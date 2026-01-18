import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
def remove_custom(self, opset: OpsetVersion) -> None:
    """Removes a custom symbolic function.

        Args:
            opset: The opset version of the custom function to remove.
        """
    if not self._functions.overridden(opset):
        warnings.warn(f"No custom function registered for '{self._name}' opset {opset}")
        return
    self._functions.remove_override(opset)