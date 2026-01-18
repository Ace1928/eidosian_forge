from __future__ import annotations
import os
import sys
from typing import Any, Iterable
import numpy as np
import onnx
import onnx.external_data_helper as ext_data
import onnx.helper
import onnx.onnx_cpp2py_export.checker as c_checker
def set_large_initializers(self, large_initializers: dict[str, np.ndarray]):
    """Adds all large tensors (not stored in the model)."""
    for k in large_initializers:
        if not self.is_in_memory_external_initializer(k):
            raise ValueError(f"The location {k!r} must start with '#' to be ignored by check model.")
    self.large_initializers = large_initializers