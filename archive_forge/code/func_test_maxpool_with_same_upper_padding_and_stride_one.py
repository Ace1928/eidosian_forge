from __future__ import annotations
import itertools
import unittest
from typing import Any, Sequence
import numpy as np
import pytest
from parameterized import parameterized
import onnx.shape_inference
from onnx import (
from onnx.defs import (
from onnx.helper import (
from onnx.parser import parse_graph
def test_maxpool_with_same_upper_padding_and_stride_one(self) -> None:
    graph = self._make_graph([('X', TensorProto.FLOAT, (5, 3, 4, 4))], [make_node('MaxPool', ['X'], ['Y'], auto_pad='SAME_UPPER', kernel_shape=[2, 2], strides=[1, 1])], [])
    self._assert_inferred(graph, [make_tensor_value_info('Y', TensorProto.FLOAT, (5, 3, 4, 4))])