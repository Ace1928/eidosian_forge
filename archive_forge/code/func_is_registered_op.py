import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
def is_registered_op(self, name: str, version: int) -> bool:
    """Returns whether the given op is registered for the given opset version."""
    functions = self.get_function_group(name)
    if functions is None:
        return False
    return functions.get(version) is not None