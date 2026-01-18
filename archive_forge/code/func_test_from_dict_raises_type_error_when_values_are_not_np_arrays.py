import unittest
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import helper, numpy_helper
def test_from_dict_raises_type_error_when_values_are_not_np_arrays(self):
    with self.assertRaises(TypeError):
        numpy_helper.from_dict({0: 0.1, 1: 0.9})