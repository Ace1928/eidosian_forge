import operator
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.intrinsic.quantized as nniq
import torch.ao.nn.intrinsic.quantized.dynamic as nniqd
import torch.ao.nn.intrinsic.qat as nniqat
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.ao.nn.qat.dynamic as nnqatd
from torch.ao.quantization.backend_config import get_native_backend_config
import torch.ao.quantization.fx._lower_to_native_backend as \
import torch.ao.quantization.quantization_mappings as quantization_mappings
from .ns_types import NSNodeTargetType
from typing import Callable, Dict, List, Optional, Set, Tuple
def get_base_name_to_sets_of_related_ops() -> Dict[str, Set[NSNodeTargetType]]:
    sets_of_related_ops: List[Set[NSNodeTargetType]] = [{nn.Conv1d}, {nn.Conv2d}, {nn.Conv3d}, {F.conv1d}, {F.conv2d}, {F.conv3d}, {nn.Linear}, {F.linear}, {nn.AvgPool1d, torch.avg_pool1d}, {nn.AvgPool2d, torch._C._nn.avg_pool2d}, {nn.AvgPool3d, torch._C._nn.avg_pool3d}, {nn.AdaptiveAvgPool1d, F.adaptive_avg_pool1d}, {nn.AdaptiveAvgPool2d, F.adaptive_avg_pool2d}, {nn.AdaptiveAvgPool3d, F.adaptive_avg_pool3d}, {nn.LSTM}, {torch.add, operator.add}, {torch.cat}, {torch.mul, operator.mul}, {F.relu, nn.ReLU, 'relu', 'relu_', torch.relu}, {nn.MaxPool1d, F.max_pool1d}, {nn.MaxPool2d, F.max_pool2d}, {nn.MaxPool3d, F.max_pool3d}, {torch.sigmoid, 'sigmoid', 'sigmoid_', nn.Sigmoid, F.sigmoid}, {nn.BatchNorm2d}, {nn.BatchNorm3d}, {nn.ConvTranspose1d}, {nn.ConvTranspose2d}, {nn.ConvTranspose3d}, {F.conv_transpose1d}, {F.conv_transpose2d}, {F.conv_transpose3d}, {nn.ELU}, {nn.Embedding}, {nn.EmbeddingBag}, {nn.GroupNorm}, {nn.Hardswish}, {nn.InstanceNorm1d}, {nn.InstanceNorm2d}, {nn.InstanceNorm3d}, {nn.LayerNorm}, {nn.LeakyReLU}, {nn.ReLU6, F.relu6}, {F.elu}, {F.hardswish}, {F.group_norm}, {F.instance_norm}, {F.layer_norm}, {F.leaky_relu}, {nn.SiLU, F.silu}, {nn.Mish, F.mish}, {nn.Tanh, F.tanh, torch.tanh, 'tanh_', 'tanh'}, {'hardsigmoid_', 'hardsigmoid', F.hardsigmoid, nn.Hardsigmoid}, {nn.Hardtanh, F.hardtanh, F.hardtanh_}, {operator.floordiv}, {torch.unsqueeze}, {torch.stack}, {torch.squeeze}, {torch.sort}, {torch.repeat_interleave}, {torch.min}, {torch.mean}, {torch.max}, {torch.transpose}, {torch.flatten}, {torch.clamp}, {torch.chunk}, {torch.nn.functional.interpolate}, {nn.Dropout}, {F.dropout}, {torch.matmul}, {nn.Softmax}, {nn.PReLU, nnq.PReLU}, {F.prelu, toq.prelu}, {nn.PixelShuffle}, {F.pixel_shuffle}, {nn.PixelUnshuffle}, {F.pixel_unshuffle}, {torch.narrow}]
    backend_config = get_native_backend_config()
    new_connections: List[Tuple[Callable, Callable]] = [(nn.Linear, nn.modules.linear.NonDynamicallyQuantizableLinear)]
    for pattern, config in backend_config._pattern_complex_format_to_config.items():
        first_element = pattern
        while isinstance(first_element, (list, tuple)):
            first_element = first_element[-1]
        if config.fused_module is not None:
            new_connections.append((first_element, config.fused_module))
        if config.qat_module is not None:
            new_connections.append((first_element, config.qat_module))
        if config.reference_quantized_module is not None:
            new_connections.append((first_element, config.reference_quantized_module))
    for source_to_target in (_lower_to_native_backend.STATIC_LOWER_MODULE_MAP, _lower_to_native_backend.DYNAMIC_LOWER_MODULE_MAP, _lower_to_native_backend.WEIGHT_ONLY_LOWER_MODULE_MAP, _lower_to_native_backend.SPECIAL_PATTERN_LOWER_MODULE_MAP):
        for source, target in source_to_target.items():
            new_connections.append((source, target))
    for source_to_double_target in (_lower_to_native_backend.STATIC_LOWER_FUSED_MODULE_MAP, _lower_to_native_backend.STATIC_LOWER_FUSED_MODULE_TWO_INPUTS_MAP, _lower_to_native_backend.DYNAMIC_LOWER_FUSED_MODULE_MAP):
        for source, (target1, target2) in source_to_double_target.items():
            new_connections.append((source, target1))
            new_connections.append((source, target2))
    for source, (target1, target2) in _lower_to_native_backend.STATIC_LOWER_FUNCTIONAL_MAP.items():
        new_connections.append((source, target1))
        new_connections.append((source, target2))
    for source_to_target in (_lower_to_native_backend.QBIN_OP_MAPPING, _lower_to_native_backend.QBIN_RELU_OP_MAPPING, quantization_mappings.DEFAULT_FLOAT_TO_QUANTIZED_OPERATOR_MAPPINGS):
        for source, target in source_to_target.items():
            new_connections.append((source, target))
    for source_to_target in (quantization_mappings.DEFAULT_DYNAMIC_QUANT_MODULE_MAPPINGS,):
        for source, target in source_to_target.items():
            new_connections.append((source, target))
    for item1, item2 in new_connections:
        for set_of_related_ops in sets_of_related_ops:
            if item1 in set_of_related_ops or item2 in set_of_related_ops:
                set_of_related_ops.add(item1)
                set_of_related_ops.add(item2)
                break
    base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]] = {}
    counter = 0
    for set_of_related_ops in sets_of_related_ops:
        base_name = str(counter)
        counter += 1
        base_name_to_sets_of_related_ops[base_name] = set_of_related_ops
    return base_name_to_sets_of_related_ops