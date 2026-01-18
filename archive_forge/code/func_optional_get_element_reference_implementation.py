from typing import Any, Optional
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
def optional_get_element_reference_implementation(optional: Optional[Any]) -> Any:
    assert optional is not None
    return optional