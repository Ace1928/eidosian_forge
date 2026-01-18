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
def test_batch_norm_train_no_shape(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, None), ('scale', TensorProto.FLOAT, None), ('b', TensorProto.FLOAT, None), ('input_mean', TensorProto.FLOAT, ('C',)), ('input_var', TensorProto.FLOAT, ('C',))], [make_node('BatchNormalization', ['x', 'scale', 'b', 'input_mean', 'input_var'], ['out', 'running_mean', 'running_var'], training_mode=1)], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, None), make_tensor_value_info('running_mean', TensorProto.FLOAT, ('C',)), make_tensor_value_info('running_var', TensorProto.FLOAT, ('C',))])