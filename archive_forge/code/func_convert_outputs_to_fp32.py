import pickle
import warnings
from functools import update_wrapper, wraps
from typing import Any, Mapping
import torch
from ..state import PartialState
from .constants import TORCH_DISTRIBUTED_OPERATION_TYPES
from .dataclasses import DistributedType, TensorInformation
from .imports import (
def convert_outputs_to_fp32(model_forward):
    model_forward = ConvertOutputsToFp32(model_forward)

    def forward(*args, **kwargs):
        return model_forward(*args, **kwargs)
    forward.__wrapped__ = model_forward
    return forward