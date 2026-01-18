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
def checkpointing(self, export_native_model_files: Optional[bool]=NotProvided, checkpoint_trainable_policies_only: Optional[bool]=NotProvided) -> 'AlgorithmConfig':
    """Sets the config's checkpointing settings.

        Args:
            export_native_model_files: Whether an individual Policy-
                or the Algorithm's checkpoints also contain (tf or torch) native
                model files. These could be used to restore just the NN models
                from these files w/o requiring RLlib. These files are generated
                by calling the tf- or torch- built-in saving utility methods on
                the actual models.
            checkpoint_trainable_policies_only: Whether to only add Policies to the
                Algorithm checkpoint (in sub-directory "policies/") that are trainable
                according to the `is_trainable_policy` callable of the local worker.

        Returns:
            This updated AlgorithmConfig object.
        """
    if export_native_model_files is not NotProvided:
        self.export_native_model_files = export_native_model_files
    if checkpoint_trainable_policies_only is not NotProvided:
        self.checkpoint_trainable_policies_only = checkpoint_trainable_policies_only
    return self