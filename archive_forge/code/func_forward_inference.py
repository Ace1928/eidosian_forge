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
@check_input_specs('_input_specs_inference')
@check_output_specs('_output_specs_inference')
def forward_inference(self, batch: SampleBatchType, **kwargs) -> Mapping[str, Any]:
    """Forward-pass during evaluation, called from the sampler.

        This method should not be overriden to implement a custom forward inference
        method. Instead, override the _forward_inference method.

        Args:
            batch: The input batch. This input batch should comply with
                input_specs_inference().
            **kwargs: Additional keyword arguments.

        Returns:
            The output of the forward pass. This output should comply with the
            ouptut_specs_inference().
        """
    return self._forward_inference(batch, **kwargs)