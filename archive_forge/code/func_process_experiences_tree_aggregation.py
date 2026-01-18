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
def process_experiences_tree_aggregation(self, worker_to_sample_batches_refs: List[Tuple[int, ObjectRef]]) -> List[SampleBatchType]:
    """Process sample batches using tree aggregation workers.

        Args:
            worker_to_sample_batches_refs: List of (worker_id, sample_batch_ref)

        NOTE: This will provide speedup when sample batches have been compressed,
        and the decompression can happen on the aggregation workers in parallel to
        the training.

        Returns:
            Batches that have been processed by the mixin buffers on the aggregation
            workers.

        """

    def _process_episodes(actor, batch):
        return actor.process_episodes(ray.get(batch))
    for _, batch in worker_to_sample_batches_refs:
        assert isinstance(batch, ObjectRef), f'For efficiency, process_experiences_tree_aggregation should be given ObjectRefs instead of {type(batch)}.'
        aggregator_id = random.choice(self._aggregator_actor_manager.healthy_actor_ids())
        calls_placed = self._aggregator_actor_manager.foreach_actor_async(partial(_process_episodes, batch=batch), remote_actor_ids=[aggregator_id])
        if calls_placed <= 0:
            self._counters['num_times_no_aggregation_worker_available'] += 1
    waiting_processed_sample_batches: RemoteCallResults = self._aggregator_actor_manager.fetch_ready_async_reqs(timeout_seconds=self._timeout_s_aggregator_manager)
    handle_remote_call_result_errors(waiting_processed_sample_batches, self.config.ignore_worker_failures)
    return [b.get() for b in waiting_processed_sample_batches.ignore_errors()]