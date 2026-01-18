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
def test_batch_norm_train_dim_param(self) -> None:
    graph = self._make_graph([('x', TensorProto.FLOAT, (3, 'C', 5, 6, 7)), ('scale', TensorProto.FLOAT, ('C',)), ('b', TensorProto.FLOAT, ('C',)), ('input_mean', TensorProto.FLOAT, ('C',)), ('input_var', TensorProto.FLOAT, ('C',))], [make_node('BatchNormalization', ['x', 'scale', 'b', 'input_mean', 'input_var'], ['out', 'output_mean', 'output_var'], training_mode=1)], [])
    self._assert_inferred(graph, [make_tensor_value_info('out', TensorProto.FLOAT, (3, 'C', 5, 6, 7)), make_tensor_value_info('output_mean', TensorProto.FLOAT, ('C',)), make_tensor_value_info('output_var', TensorProto.FLOAT, ('C',))])