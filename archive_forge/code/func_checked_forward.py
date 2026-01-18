import abc
import logging
from typing import Tuple, Union
import numpy as np
from ray.rllib.core.models.base import Model
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.specs.checker import (
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType
from ray.util import log_once
@check_input_specs('input_specs', only_check_on_retry=False)
@check_output_specs('output_specs')
def checked_forward(self, input_data, **kwargs):
    return self._forward(input_data, **kwargs)