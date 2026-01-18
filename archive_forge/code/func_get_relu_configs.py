import operator
import torch
from torch.ao.quantization.backend_config import (
from typing import List
def get_relu_configs():
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]
    backend_pattern_configs.append(BackendPatternConfig(torch.ops.aten.relu.default).set_observation_type(observation_type).set_dtype_configs(dtype_configs))
    return backend_pattern_configs