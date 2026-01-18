from typing import Dict
import torch
from . import __version__, _cpp_lib, _is_opensource, _is_triton_available, ops
from .ops.common import OPERATORS_REGISTRY
from .profiler.profiler_dcgm import DCGM_PROFILER_AVAILABLE
def get_features_status() -> Dict[str, str]:
    features = {}
    for op in OPERATORS_REGISTRY:
        status_str = 'available' if op.is_available() else 'unavailable'
        features[f'{op.OPERATOR_CATEGORY}.{op.NAME}'] = status_str
    for k, v in ops.swiglu_op._info().items():
        features[f'swiglu.{k}'] = v
    features['is_triton_available'] = str(_is_triton_available())
    return features