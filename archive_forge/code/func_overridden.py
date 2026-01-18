import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
def overridden(self, key: _K) -> bool:
    """Checks if a key-value pair is overridden."""
    return key in self._overrides