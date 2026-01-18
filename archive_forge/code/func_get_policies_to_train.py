import copy
import importlib.util
import logging
import os
import platform
import threading
from collections import defaultdict
from types import FunctionType
from typing import (
import numpy as np
import tree  # pip install dm_tree
from gymnasium.spaces import Discrete, MultiDiscrete, Space
import ray
from ray import ObjectRef
from ray import cloudpickle as pickle
from ray.rllib.connectors.util import (
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env.base_env import BaseEnv, convert_to_base_env
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib.env.external_multi_agent_env import ExternalMultiAgentEnv
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind
from ray.rllib.evaluation.metrics import RolloutMetrics
from ray.rllib.evaluation.sampler import SyncSampler
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from ray.rllib.offline import (
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.policy.policy_map import PolicyMap
from ray.rllib.policy.sample_batch import (
from ray.rllib.policy.torch_policy import TorchPolicy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils import check_env, force_list
from ray.rllib.utils.annotations import DeveloperAPI, override
from ray.rllib.utils.debug import summarize, update_global_seed_if_necessary
from ray.rllib.utils.deprecation import DEPRECATED_VALUE, deprecation_warning
from ray.rllib.utils.error import ERR_MSG_NO_GPUS, HOWTO_CHANGE_CONFIG
from ray.rllib.utils.filter import Filter, NoFilter, get_filter
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.from_config import from_config
from ray.rllib.utils.policy import create_policy_for_framework, validate_policy_id
from ray.rllib.utils.sgd import do_minibatch_sgd
from ray.rllib.utils.tf_run_builder import _TFRunBuilder
from ray.rllib.utils.tf_utils import get_gpu_devices as get_tf_gpu_devices
from ray.rllib.utils.tf_utils import get_tf_eager_cls_if_necessary
from ray.rllib.utils.typing import (
from ray.tune.registry import registry_contains_input, registry_get_input
from ray.util.annotations import PublicAPI
from ray.util.debug import disable_log_once_globally, enable_periodic_logging, log_once
from ray.util.iter import ParallelIteratorWorker
@PublicAPI(stability='alpha')
def get_policies_to_train(self, batch: Optional[SampleBatchType]=None) -> Set[PolicyID]:
    """Returns all policies-to-train, given an optional batch.

        Loops through all policies currently in `self.policy_map` and checks
        the return value of `self.is_policy_to_train(pid, batch)`.

        Args:
            batch: An optional SampleBatchType for the
                `self.is_policy_to_train(pid, [batch]?)` check.

        Returns:
            The set of currently trainable policy IDs, given the optional
            `batch`.
        """
    return {pid for pid in self.policy_map.keys() if self.is_policy_to_train is None or self.is_policy_to_train(pid, batch)}