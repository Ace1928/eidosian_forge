import itertools
import random
import struct
import unittest
from typing import Any, List, Tuple
import numpy as np
import parameterized
import pytest
import version_utils
from onnx import (
from onnx.reference.op_run import to_array_extended
def test_make_tensor_value_info(self) -> None:
    vi = helper.make_tensor_value_info('X', TensorProto.FLOAT, (2, 4))
    checker.check_value_info(vi)
    vi = helper.make_tensor_value_info('Y', TensorProto.FLOAT, ())
    checker.check_value_info(vi)