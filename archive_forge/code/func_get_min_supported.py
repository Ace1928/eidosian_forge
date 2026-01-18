import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
def get_min_supported(self) -> OpsetVersion:
    """Returns the lowest built-in opset version supported by the function."""
    return min(self._functions)