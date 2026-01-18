from collections import Counter
import copy
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete, MultiBinary
from gymnasium.spaces import Dict as GymDict
from gymnasium.spaces import Tuple as GymTuple
import inspect
import logging
import numpy as np
import os
import pprint
import random
import re
import time
import tree  # pip install dm_tree
from typing import (
import yaml
import ray
from ray import air, tune
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.typing import ResultDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.tune import CLIReporter, run_experiments
def run_learning_tests_from_yaml(yaml_files: List[str], *, framework: Optional[str]=None, max_num_repeats: int=2, use_pass_criteria_as_stop: bool=True, smoke_test: bool=False) -> Dict[str, Any]:
    """Runs the given experiments in yaml_files and returns results dict.

    Args:
        framework: The framework to use for running this test. If None,
            run the test on all frameworks.
        yaml_files: List of yaml file names.
        max_num_repeats: How many times should we repeat a failed
            experiment?
        use_pass_criteria_as_stop: Configure the Trial so that it stops
            as soon as pass criterias are met.
        smoke_test: Whether this is just a smoke-test. If True,
            set time_total_s to 5min and don't early out due to rewards
            or timesteps reached.

    Returns:
        A results dict mapping strings (e.g. "time_taken", "stats", "passed") to
            the respective stats/values.
    """
    print('Will run the following yaml files:')
    for yaml_file in yaml_files:
        print('->', yaml_file)
    all_trials = []
    experiments = {}
    checks = {}
    stats = {}
    start_time = time.monotonic()

    def should_check_eval(experiment):
        return experiment['config'].get('evaluation_interval', None) is not None
    for yaml_file in yaml_files:
        tf_experiments = yaml.safe_load(open(yaml_file).read())
        for k, e in tf_experiments.items():
            if framework is not None:
                frameworks = [framework]
            elif 'frameworks' in e:
                frameworks = e['frameworks']
            else:
                frameworks = ['tf', 'torch']
            e.pop('frameworks', None)
            e['stop'] = e['stop'] if 'stop' in e else {}
            e['pass_criteria'] = e['pass_criteria'] if 'pass_criteria' in e else {}
            check_eval = should_check_eval(e)
            episode_reward_key = 'sampler_results/episode_reward_mean' if not check_eval else 'evaluation/sampler_results/episode_reward_mean'
            if smoke_test:
                e['stop']['time_total_s'] = 0
            elif use_pass_criteria_as_stop:
                min_reward = e.get('pass_criteria', {}).get(episode_reward_key)
                if min_reward is not None:
                    e['stop'][episode_reward_key] = min_reward
            for framework in frameworks:
                k_ = k + '-' + framework
                ec = copy.deepcopy(e)
                ec['config']['framework'] = framework
                if framework == 'tf2':
                    ec['config']['eager_tracing'] = True
                checks[k_] = {'min_reward': ec['pass_criteria'].get(episode_reward_key, 0.0), 'min_throughput': ec['pass_criteria'].get('timesteps_total', 0.0) / (ec['stop'].get('time_total_s', 1.0) or 1.0), 'time_total_s': ec['stop'].get('time_total_s'), 'failures': 0, 'passed': False}
                ec.pop('pass_criteria', None)
                experiments[k_] = ec
    experiments_to_run = experiments.copy()
    release_test_storage_path = '/mnt/cluster_storage'
    if os.path.exists(release_test_storage_path):
        for k, e in experiments_to_run.items():
            e['storage_path'] = release_test_storage_path
    try:
        ray.init(address='auto')
    except ConnectionError:
        ray.init()
    for i in range(max_num_repeats):
        if len(experiments_to_run) == 0:
            print('All experiments finished.')
            break
        print(f'Starting learning test iteration {i}...')
        print('== Test config ==')
        print(yaml.dump(experiments_to_run))
        trials = run_experiments(experiments_to_run, resume=False, verbose=2, progress_reporter=CLIReporter(metric_columns={'training_iteration': 'iter', 'time_total_s': 'time_total_s', NUM_ENV_STEPS_SAMPLED: 'ts (sampled)', NUM_ENV_STEPS_TRAINED: 'ts (trained)', 'episodes_this_iter': 'train_episodes', 'episode_reward_mean': 'reward_mean', 'evaluation/episode_reward_mean': 'eval_reward_mean'}, parameter_columns=['framework'], sort_by_metric=True, max_report_frequency=30))
        all_trials.extend(trials)
        for experiment in experiments_to_run.copy():
            print(f'Analyzing experiment {experiment} ...')
            trials_for_experiment = []
            for t in trials:
                trial_exp = re.sub('.+/([^/]+)$', '\\1', t.local_dir)
                if trial_exp == experiment:
                    trials_for_experiment.append(t)
            print(f' ... Trials: {trials_for_experiment}.')
            check_eval = should_check_eval(experiments[experiment])
            if any((t.status == 'ERROR' for t in trials_for_experiment)):
                print(' ... ERROR.')
                checks[experiment]['failures'] += 1
            elif smoke_test:
                print(' ... SMOKE TEST (mark ok).')
                checks[experiment]['passed'] = True
                del experiments_to_run[experiment]
            else:
                if check_eval:
                    episode_reward_mean = np.mean([t.metric_analysis['evaluation/sampler_results/episode_reward_mean']['max'] for t in trials_for_experiment])
                else:
                    episode_reward_mean = np.mean([t.metric_analysis['sampler_results/episode_reward_mean']['max'] for t in trials_for_experiment])
                desired_reward = checks[experiment]['min_reward']
                timesteps_total = np.mean([t.last_result['timesteps_total'] for t in trials_for_experiment])
                total_time_s = np.mean([t.last_result['time_total_s'] for t in trials_for_experiment])
                throughput = timesteps_total / (total_time_s or 1.0)
                desired_throughput = None
                stats[experiment] = {'episode_reward_mean': float(episode_reward_mean), 'throughput': float(throughput) if throughput is not None else 0.0}
                print(f' ... Desired reward={desired_reward}; desired throughput={desired_throughput}')
                if desired_reward and episode_reward_mean < desired_reward or (desired_throughput and throughput < desired_throughput):
                    print(f' ... Not successful: Actual reward={episode_reward_mean}; actual throughput={throughput}')
                    checks[experiment]['failures'] += 1
                else:
                    print(f' ... Successful: (mark ok). Actual reward={episode_reward_mean}; actual throughput={throughput}')
                    checks[experiment]['passed'] = True
                    del experiments_to_run[experiment]
    ray.shutdown()
    time_taken = time.monotonic() - start_time
    result = {'time_taken': float(time_taken), 'trial_states': dict(Counter([trial.status for trial in all_trials])), 'last_update': float(time.time()), 'stats': stats, 'passed': [k for k, exp in checks.items() if exp['passed']], 'not_passed': [k for k, exp in checks.items() if not exp['passed']], 'failures': {k: exp['failures'] for k, exp in checks.items() if exp['failures'] > 0}}
    return result