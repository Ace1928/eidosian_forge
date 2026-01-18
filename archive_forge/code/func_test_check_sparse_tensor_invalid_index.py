import os
import tempfile
import unittest
from typing import Sequence
import numpy as np
import onnx.defs
import onnx.parser
from onnx import (
def test_check_sparse_tensor_invalid_index(self) -> None:
    sparse = self.make_sparse([100], [13, 17, 19], [3], [9, 27, 181])
    self.assertRaises(checker.ValidationError, checker.check_sparse_tensor, sparse)