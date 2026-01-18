from collections import OrderedDict
from typing import Any, Mapping, Optional
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging
def custom_unfold(input, dimension, size, step):
    """Custom torch.Tensor.unfold implementation to enable the export to ONNX."""
    import torch
    shape = input.size()
    rank = len(shape)
    sizedim = shape[dimension]
    low_indices = torch.arange(0, sizedim, step)
    min_length = torch.div(sizedim - size, step, rounding_mode='floor') + 1
    indices = torch.arange(size) + low_indices[:min_length][:, None]
    s = [slice(None)] * rank
    s[dimension] = indices
    sliced = input[s]
    perm = list(range(0, rank + 1))
    perm.append(perm.pop(dimension + 1))
    return sliced.permute(perm)