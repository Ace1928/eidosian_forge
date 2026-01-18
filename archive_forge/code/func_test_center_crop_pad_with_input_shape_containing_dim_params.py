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
def test_center_crop_pad_with_input_shape_containing_dim_params(self):
    graph = self._make_graph([('input_data', TensorProto.FLOAT, (20, 'W', 3)), ('shape', TensorProto.INT64, (2,))], [make_node('CenterCropPad', ['input_data', 'shape'], ['y'], axes=[0, 1])], [], initializer=[make_tensor('shape', TensorProto.INT64, (2,), (10, 8))])
    self._assert_inferred(graph, [make_tensor_value_info('y', TensorProto.FLOAT, (10, 8, 3))], opset_imports=[helper.make_opsetid(ONNX_DOMAIN, 18)])