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
def get_learner_group_config(self, module_spec: ModuleSpec) -> LearnerGroupConfig:
    if not self._is_frozen:
        raise ValueError('Cannot call `get_learner_group_config()` on an unfrozen AlgorithmConfig! Please call `AlgorithmConfig.freeze()` first.')
    config = LearnerGroupConfig().module(module_spec).learner(learner_class=self.learner_class, learner_hyperparameters=self.get_learner_hyperparameters()).resources(num_learner_workers=self.num_learner_workers, num_cpus_per_learner_worker=self.num_cpus_per_learner_worker if not self.num_gpus_per_learner_worker else 0, num_gpus_per_learner_worker=self.num_gpus_per_learner_worker, local_gpu_idx=self.local_gpu_idx)
    if self.framework_str == 'torch':
        config.framework(torch_compile=self.torch_compile_learner, torch_compile_cfg=self.get_torch_compile_learner_config(), torch_compile_what_to_compile=self.torch_compile_learner_what_to_compile)
    elif self.framework_str == 'tf2':
        config.framework(eager_tracing=self.eager_tracing)
    return config