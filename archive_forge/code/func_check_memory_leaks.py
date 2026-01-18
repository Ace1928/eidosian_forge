from collections import defaultdict
import numpy as np
import tree  # pip install dm_tree
from typing import DefaultDict, List, Optional, Set
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.util.debug import _test_some_code_for_memory_leaks, Suspect
@DeveloperAPI
def check_memory_leaks(algorithm, to_check: Optional[Set[str]]=None, repeats: Optional[int]=None, max_num_trials: int=3) -> DefaultDict[str, List[Suspect]]:
    """Diagnoses the given Algorithm for possible memory leaks.

    Isolates single components inside the Algorithm's local worker, e.g. the env,
    policy, etc.. and calls some of their methods repeatedly, while checking
    the memory footprints and keeping track of which lines in the code add
    un-GC'd items to memory.

    Args:
        algorithm: The Algorithm instance to test.
        to_check: Set of strings to indentify components to test. Allowed strings
            are: "env", "policy", "model", "rollout_worker". By default, check all
            of these.
        repeats: Number of times the test code block should get executed (per trial).
            If a trial fails, a new trial may get started with a larger number of
            repeats: actual_repeats = `repeats` * (trial + 1) (1st trial == 0).
        max_num_trials: The maximum number of trials to run each check for.

    Raises:
        A defaultdict(list) with keys being the `to_check` strings and values being
        lists of Suspect instances that were found.
    """
    local_worker = algorithm.workers.local_worker()
    to_check = to_check or {'env', 'model', 'policy', 'rollout_worker'}
    results_per_category = defaultdict(list)
    if 'env' in to_check:
        assert local_worker.async_env is not None, "ERROR: Cannot test 'env' since given Algorithm does not have one in its local worker. Try setting `create_env_on_driver=True`."
        env = local_worker.async_env.get_sub_environments()[0]
        action_space = env.action_space
        action_sample = action_space.sample()

        def code():
            ts = 0
            env.reset()
            while True:
                _, _, done, _, _ = env.step(action_sample)
                ts += 1
                if done:
                    break
        test = _test_some_code_for_memory_leaks(desc='Looking for leaks in env, running through episodes.', init=None, code=code, repeats=repeats or 200, max_num_trials=max_num_trials)
        if test:
            results_per_category['env'].extend(test)
    if 'policy' in to_check:
        policy = local_worker.policy_map[DEFAULT_POLICY_ID]
        obs = tree.map_structure(lambda s: np.stack([s] * 10, axis=0), policy.observation_space.sample())
        print('Looking for leaks in Policy')

        def code():
            policy.compute_actions_from_input_dict({'obs': obs})
        test = _test_some_code_for_memory_leaks(desc='Calling `compute_actions_from_input_dict()`.', init=None, code=code, repeats=repeats or 400, max_num_trials=max_num_trials)
        if test:
            results_per_category['policy'].extend(test)
        if not policy.config.get('_enable_new_api_stack', False):
            dummy_batch = policy._get_dummy_batch_from_view_requirements(batch_size=16)
            test = _test_some_code_for_memory_leaks(desc='Calling `learn_on_batch()`.', init=None, code=lambda: policy.learn_on_batch(dummy_batch), repeats=repeats or 100, max_num_trials=max_num_trials)
            if test:
                results_per_category['policy'].extend(test)
    if 'model' in to_check:
        policy = local_worker.policy_map[DEFAULT_POLICY_ID]
        obs = tree.map_structure(lambda s: s[None], policy.observation_space.sample())
        print('Looking for leaks in Model')
        test = _test_some_code_for_memory_leaks(desc='Calling `[model]()`.', init=None, code=lambda: policy.model({SampleBatch.OBS: obs}), repeats=repeats or 400, max_num_trials=max_num_trials)
        if test:
            results_per_category['model'].extend(test)
    if 'rollout_worker' in to_check:
        print('Looking for leaks in local RolloutWorker')

        def code():
            local_worker.sample()
            local_worker.get_metrics()
        test = _test_some_code_for_memory_leaks(desc='Calling `sample()` and `get_metrics()`.', init=None, code=code, repeats=repeats or 50, max_num_trials=max_num_trials)
        if test:
            results_per_category['rollout_worker'].extend(test)
    if 'learner' in to_check and algorithm.config.get('_enable_new_api_stack', False):
        learner_group = algorithm.learner_group
        assert learner_group._is_local, 'This test will miss leaks hidden in remote workers. Please make sure that there is a local learner inside the learner group for this test.'
        dummy_batch = algorithm.get_policy()._get_dummy_batch_from_view_requirements(batch_size=16).as_multi_agent()
        print('Looking for leaks in Learner')

        def code():
            learner_group.update(dummy_batch)
        test = _test_some_code_for_memory_leaks(desc='Calling `LearnerGroup.update()`.', init=None, code=code, repeats=repeats or 400, max_num_trials=max_num_trials)
        if test:
            results_per_category['learner'].extend(test)
    return results_per_category