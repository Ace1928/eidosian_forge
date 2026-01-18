import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
def get_function_group(self, name: str) -> Optional[_SymbolicFunctionGroup]:
    """Returns the function group for the given name."""
    return self._registry.get(name)