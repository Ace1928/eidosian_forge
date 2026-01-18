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
def process_experiences_directly(self, worker_to_sample_batches: List[Tuple[int, SampleBatch]]) -> List[SampleBatchType]:
    """Process sample batches directly on the driver, for training.

        Args:
            worker_to_sample_batches: List of (worker_id, sample_batch) tuples.

        Returns:
            Batches that have been processed by the mixin buffer.

        """
    batches = [b for _, b in worker_to_sample_batches]
    processed_batches = []
    for batch in batches:
        assert not isinstance(batch, ObjectRef), 'process_experiences_directly can not handle ObjectRefs. '
        batch = batch.decompress_if_needed()
        self.local_mixin_buffer.add(batch)
        batch = self.local_mixin_buffer.replay(_ALL_POLICIES)
        if batch:
            processed_batches.append(batch)
    return processed_batches