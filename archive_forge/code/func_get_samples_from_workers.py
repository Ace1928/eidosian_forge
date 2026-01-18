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
def get_samples_from_workers(self, return_object_refs: Optional[bool]=False) -> List[Tuple[int, Union[ObjectRef, SampleBatchType]]]:
    """Get samples from rollout workers for training.

        Args:
            return_object_refs: If True, return ObjectRefs instead of the samples
                directly. This is useful when using aggregator workers so that data
                collected on rollout workers is directly de referenced on the aggregator
                workers instead of first in the driver and then on the aggregator
                workers.

        Returns:
            a list of tuples of (worker_index, sample batch or ObjectRef to a sample
                batch)

        """
    with self._timers[SAMPLE_TIMER]:
        if self.workers.num_healthy_remote_workers() > 0:
            self.workers.foreach_worker_async(lambda worker: worker.sample(), healthy_only=True)
            sample_batches: List[Tuple[int, ObjectRef]] = self.workers.fetch_ready_async_reqs(timeout_seconds=self._timeout_s_sampler_manager, return_obj_refs=return_object_refs)
        elif self.workers.local_worker() and self.workers.local_worker().async_env is not None:
            sample_batch = self.workers.local_worker().sample()
            if return_object_refs:
                sample_batch = ray.put(sample_batch)
            sample_batches = [(0, sample_batch)]
        else:
            return []
    return sample_batches