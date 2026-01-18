import abc
import datetime
import json
import pathlib
from dataclasses import dataclass
from typing import Mapping, Any, TYPE_CHECKING, Optional, Type, Dict, Union
import gymnasium as gym
import tree
import ray
from ray.rllib.utils.annotations import (
from ray.rllib.utils.typing import ViewRequirementsDict
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic
from ray.rllib.core.models.base import STATE_IN, STATE_OUT
from ray.rllib.policy.policy import get_gym_space_from_struct_of_tensors
from ray.rllib.policy.view_requirement import ViewRequirement
from ray.rllib.core.models.specs.typing import SpecType
from ray.rllib.core.models.specs.checker import (
from ray.rllib.models.distributions import Distribution
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID, SampleBatch
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.numpy import convert_to_numpy
from ray.rllib.utils.typing import SampleBatchType
from ray.rllib.utils.serialization import (
def input_specs_inference(self) -> SpecType:
    """Returns the input specs of the forward_inference method."""
    return self._default_input_specs()