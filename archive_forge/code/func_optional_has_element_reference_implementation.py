from typing import Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def optional_has_element_reference_implementation(optional: Optional[np.ndarray]) -> np.ndarray:
    if optional is None:
        return np.array(False)
    else:
        return np.array(True)