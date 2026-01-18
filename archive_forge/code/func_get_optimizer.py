import abc
import json
import logging
import pathlib
from collections import defaultdict
from enum import Enum
from dataclasses import dataclass, field
from typing import (
import ray
from ray.rllib.core.learner.reduce_result_dict_fn import _reduce_mean_results
from ray.rllib.core.learner.scaling_config import LearnerGroupScalingConfig
from ray.rllib.core.rl_module.marl_module import (
from ray.rllib.core.rl_module.rl_module import (
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, MultiAgentBatch
from ray.rllib.utils.annotations import (
from ray.rllib.utils.debug import update_global_seed_if_necessary
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
from ray.rllib.utils.minibatch_utils import (
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.serialization import serialize_type
from ray.rllib.utils.typing import (
from ray.util.annotations import PublicAPI
def get_optimizer(self, module_id: ModuleID=DEFAULT_POLICY_ID, optimizer_name: str=DEFAULT_OPTIMIZER) -> Optimizer:
    """Returns the optimizer object, configured under the given module_id and name.

        If only one optimizer was registered under `module_id` (or ALL_MODULES)
        via the `self.register_optimizer` method, `optimizer_name` is assumed to be
        DEFAULT_OPTIMIZER.

        Args:
            module_id: The ModuleID for which to return the configured optimizer.
                If not provided, will assume DEFAULT_POLICY_ID.
            optimizer_name: The name of the optimizer (registered under `module_id` via
                `self.register_optimizer()`) to return. If not provided, will assume
                DEFAULT_OPTIMIZER.

        Returns:
            The optimizer object, configured under the given `module_id` and
            `optimizer_name`.
        """
    full_registration_name = module_id + '_' + optimizer_name
    assert full_registration_name in self._named_optimizers
    return self._named_optimizers[full_registration_name]