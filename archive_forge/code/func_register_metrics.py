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
def register_metrics(self, module_id: str, metrics_dict: Dict[str, Any]) -> None:
    """Registers several key/value metric pairs for loss- and gradient stats.

        Args:
            module_id: The module_id to register the metrics under. This may be
                ALL_MODULES.
            metrics_dict: A dict mapping names of metrics to be registered (below the
                given `module_id`) to the actual values of these metrics. Values might
                also be tensor vars (e.g. from within a traced tf2 function).
                These will be automatically converted to numpy values.
        """
    for key, value in metrics_dict.items():
        self.register_metric(module_id, key, value)