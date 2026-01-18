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
def set_is_policy_to_train(self, is_policy_to_train: Union[Container[PolicyID], Callable[[PolicyID, Optional[SampleBatchType]], bool]]) -> None:
    """Sets `self.is_policy_to_train()` to a new callable.

        Args:
            is_policy_to_train: A container of policy IDs to be
                trained or a callable taking PolicyID and - optionally -
                SampleBatchType and returning a bool (trainable or not?).
                If None, will keep the existing setup in place.
                Policies, whose IDs are not in the list (or for which the
                callable returns False) will not be updated.
        """
    if not callable(is_policy_to_train):
        assert isinstance(is_policy_to_train, (list, set, tuple)), 'ERROR: `is_policy_to_train`must be a [list|set|tuple] or a callable taking PolicyID and SampleBatch and returning True|False (trainable or not?).'
        pols = set(is_policy_to_train)

        def is_policy_to_train(pid, batch=None):
            return pid in pols
    self.is_policy_to_train = is_policy_to_train