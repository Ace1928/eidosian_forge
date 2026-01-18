import copy
import dataclasses
from functools import partial
import logging
import platform
import queue
import random
from typing import Callable, List, Optional, Set, Tuple, Type, Union
import numpy as np
import tree  # pip install dm_tree
import ray
from ray import ObjectRef
from ray.rllib import SampleBatch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.impala.impala_learner import (
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.evaluation.worker_set import handle_remote_call_result_errors
from ray.rllib.execution.buffers.mixin_replay_buffer import MixInMultiAgentReplayBuffer
from ray.rllib.execution.learner_thread import LearnerThread
from ray.rllib.execution.multi_gpu_learner_thread import MultiGPULearnerThread
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import concat_samples
from ray.rllib.utils.actor_manager import (
from ray.rllib.utils.actors import create_colocated_actors
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.metrics import (
from ray.rllib.utils.metrics.learner_info import LearnerInfoBuilder
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import ReplayMode
from ray.rllib.utils.replay_buffers.replay_buffer import _ALL_POLICIES
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.typing import (
from ray.tune.execution.placement_groups import PlacementGroupFactory
def process_trained_results(self) -> ResultDict:
    """Process training results that are outputed by the learner thread.

        NOTE: This method is called if self.config._enable_new_api_stack is False.

        Returns:
            Aggregated results from the learner thread after an update is completed.

        """
    num_env_steps_trained = 0
    num_agent_steps_trained = 0
    learner_infos = []
    for _ in range(self._learner_thread.outqueue.qsize()):
        env_steps, agent_steps, learner_results = self._learner_thread.outqueue.get(timeout=0.001)
        num_env_steps_trained += env_steps
        num_agent_steps_trained += agent_steps
        if learner_results:
            learner_infos.append(learner_results)
    if not learner_infos:
        final_learner_info = copy.deepcopy(self._learner_thread.learner_info)
    else:
        builder = LearnerInfoBuilder()
        for info in learner_infos:
            builder.add_learn_on_batch_results_multi_agent(info)
        final_learner_info = builder.finalize()
    self._counters[NUM_ENV_STEPS_TRAINED] += num_env_steps_trained
    self._counters[NUM_AGENT_STEPS_TRAINED] += num_agent_steps_trained
    return final_learner_info