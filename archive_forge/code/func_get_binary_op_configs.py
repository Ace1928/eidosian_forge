import operator
import torch
from torch.ao.quantization.backend_config import (
from typing import List
def get_binary_op_configs():
    binary_op_configs: List[BackendPatternConfig] = []
    dtype_configs = [weighted_op_quint8_dtype_config]
    num_tensor_args_to_observation_type_mapping = {0: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT, 1: ObservationType.OUTPUT_SHARE_OBSERVER_WITH_INPUT, 2: ObservationType.OUTPUT_USE_DIFFERENT_OBSERVER_AS_INPUT}
    for op_with_quantized_bop_scalar_variant in [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]:
        bop_patterns = [(op_with_quantized_bop_scalar_variant, torch.ops.aten.relu.default), op_with_quantized_bop_scalar_variant, (op_with_quantized_bop_scalar_variant, torch.ops.aten.relu_.default)]
        for bop_pattern in bop_patterns:
            binary_op_configs.append(BackendPatternConfig(bop_pattern).set_dtype_configs(dtype_configs)._set_num_tensor_args_to_observation_type(num_tensor_args_to_observation_type_mapping))
    return binary_op_configs