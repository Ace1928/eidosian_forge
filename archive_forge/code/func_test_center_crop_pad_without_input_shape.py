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
def test_center_crop_pad_without_input_shape(self):
    graph = self._make_graph([('input_data', TensorProto.FLOAT, (3, 2)), ('shape', TensorProto.INT64, (2,))], [make_node('CenterCropPad', ['input_data', 'shape'], ['y'])], [])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, None)], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)])