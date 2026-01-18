import random
import importlib
from typing import Any, Dict, Optional
import numpy as np
import torch
from vllm.config import DeviceConfig, ModelConfig
def set_weight_attrs(weight: torch.Tensor, weight_attrs: Optional[Dict[str, Any]]):
    """Set attributes on a weight tensor.

    This method is used to set attributes on a weight tensor. This method
    will not overwrite existing attributes.

    Args:
        weight: The weight tensor.
        weight_attrs: A dictionary of attributes to set on the weight tensor.
    """
    if weight_attrs is None:
        return
    for key, value in weight_attrs.items():
        assert not hasattr(weight, key), f'Overwriting existing tensor attribute: {key}'
        setattr(weight, key, value)