import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_HannWindow(self) -> None:
    self._test_window_function('HannWindow')