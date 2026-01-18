import operator
import torch
from torch.ao.quantization.backend_config import (
from typing import List
def get_pooling_configs():
    backend_pattern_configs = []
    observation_type = ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT
    dtype_configs = [weighted_op_quint8_dtype_config]

    def root_node_getter(node_pattern):
        getitem, maxpool, index = node_pattern
        return maxpool
    backend_pattern_configs.append(BackendPatternConfig()._set_pattern_complex_format((operator.getitem, torch.ops.aten.max_pool2d_with_indices.default, 0)).set_observation_type(observation_type).set_dtype_configs(dtype_configs)._set_root_node_getter(root_node_getter))
    return backend_pattern_configs