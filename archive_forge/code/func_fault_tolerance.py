import copy
import logging
import math
import os
import sys
from typing import (
from packaging import version
import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.core.learner.learner import LearnerHyperparameters
from ray.rllib.core.learner.learner_group_config import LearnerGroupConfig, ModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import ModuleID, SingleAgentRLModuleSpec
from ray.rllib.core.learner.learner import TorchCompileWhatToCompile
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari
from ray.rllib.evaluation.collectors.sample_collector import SampleCollector
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.rollout_worker import RolloutWorker
from ray.rllib.models import MODEL_DEFAULTS
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils import deep_update, merge_dicts
from ray.rllib.utils.annotations import (
from ray.rllib.utils.deprecation import (
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import NotProvided, from_config
from ray.rllib.utils.gym import (
from ray.rllib.utils.policy import validate_policy_id
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import (
from ray.rllib.utils.torch_utils import TORCH_COMPILE_REQUIRED_VERSION
from ray.rllib.utils.typing import (
from ray.tune.logger import Logger
from ray.tune.registry import get_trainable_cls
from ray.tune.result import TRIAL_INFO
from ray.tune.tune import _Config
def fault_tolerance(self, recreate_failed_workers: Optional[bool]=NotProvided, max_num_worker_restarts: Optional[int]=NotProvided, delay_between_worker_restarts_s: Optional[float]=NotProvided, restart_failed_sub_environments: Optional[bool]=NotProvided, num_consecutive_worker_failures_tolerance: Optional[int]=NotProvided, worker_health_probe_timeout_s: int=NotProvided, worker_restore_timeout_s: int=NotProvided):
    """Sets the config's fault tolerance settings.

        Args:
            recreate_failed_workers: Whether - upon a worker failure - RLlib will try to
                recreate the lost worker as an identical copy of the failed one. The new
                worker will only differ from the failed one in its
                `self.recreated_worker=True` property value. It will have the same
                `worker_index` as the original one. If True, the
                `ignore_worker_failures` setting will be ignored.
            max_num_worker_restarts: The maximum number of times a worker is allowed to
                be restarted (if `recreate_failed_workers` is True).
            delay_between_worker_restarts_s: The delay (in seconds) between two
                consecutive worker restarts (if `recreate_failed_workers` is True).
            restart_failed_sub_environments: If True and any sub-environment (within
                a vectorized env) throws any error during env stepping, the
                Sampler will try to restart the faulty sub-environment. This is done
                without disturbing the other (still intact) sub-environment and without
                the EnvRunner crashing.
            num_consecutive_worker_failures_tolerance: The number of consecutive times
                a rollout worker (or evaluation worker) failure is tolerated before
                finally crashing the Algorithm. Only useful if either
                `ignore_worker_failures` or `recreate_failed_workers` is True.
                Note that for `restart_failed_sub_environments` and sub-environment
                failures, the worker itself is NOT affected and won't throw any errors
                as the flawed sub-environment is silently restarted under the hood.
            worker_health_probe_timeout_s: Max amount of time we should spend waiting
                for health probe calls to finish. Health pings are very cheap, so the
                default is 1 minute.
            worker_restore_timeout_s: Max amount of time we should wait to restore
                states on recovered worker actors. Default is 30 mins.

        Returns:
            This updated AlgorithmConfig object.
        """
    if recreate_failed_workers is not NotProvided:
        self.recreate_failed_workers = recreate_failed_workers
    if max_num_worker_restarts is not NotProvided:
        self.max_num_worker_restarts = max_num_worker_restarts
    if delay_between_worker_restarts_s is not NotProvided:
        self.delay_between_worker_restarts_s = delay_between_worker_restarts_s
    if restart_failed_sub_environments is not NotProvided:
        self.restart_failed_sub_environments = restart_failed_sub_environments
    if num_consecutive_worker_failures_tolerance is not NotProvided:
        self.num_consecutive_worker_failures_tolerance = num_consecutive_worker_failures_tolerance
    if worker_health_probe_timeout_s is not NotProvided:
        self.worker_health_probe_timeout_s = worker_health_probe_timeout_s
    if worker_restore_timeout_s is not NotProvided:
        self.worker_restore_timeout_s = worker_restore_timeout_s
    return self