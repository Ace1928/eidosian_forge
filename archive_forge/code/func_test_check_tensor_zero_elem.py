import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_tensor_zero_elem(self) -> None:
    tensor = self._sample_0_elem_tensor
    checker.check_tensor(tensor)