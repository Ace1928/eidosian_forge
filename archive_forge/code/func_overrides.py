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
@classmethod
def overrides(cls, **kwargs):
    """Generates and validates a set of config key/value pairs (passed via kwargs).

        Validation whether given config keys are valid is done immediately upon
        construction (by comparing against the properties of a default AlgorithmConfig
        object of this class).
        Allows combination with a full AlgorithmConfig object to yield a new
        AlgorithmConfig object.

        Used anywhere, we would like to enable the user to only define a few config
        settings that would change with respect to some main config, e.g. in multi-agent
        setups and evaluation configs.

        .. testcode::

            from ray.rllib.algorithms.ppo import PPOConfig
            from ray.rllib.policy.policy import PolicySpec
            config = (
                PPOConfig()
                .multi_agent(
                    policies={
                        "pol0": PolicySpec(config=PPOConfig.overrides(lambda_=0.95))
                    },
                )
            )


        .. testcode::

            from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
            from ray.rllib.algorithms.ppo import PPOConfig
            config = (
                PPOConfig()
                .evaluation(
                    evaluation_num_workers=1,
                    evaluation_interval=1,
                    evaluation_config=AlgorithmConfig.overrides(explore=False),
                )
            )

        Returns:
            A dict mapping valid config property-names to values.

        Raises:
            KeyError: In case a non-existing property name (kwargs key) is being
            passed in. Valid property names are taken from a default AlgorithmConfig
            object of `cls`.
        """
    default_config = cls()
    config_overrides = {}
    for key, value in kwargs.items():
        if not hasattr(default_config, key):
            raise KeyError(f'Invalid property name {key} for config class {cls.__name__}!')
        key = cls._translate_special_keys(key, warn_deprecated=True)
        config_overrides[key] = value
    return config_overrides