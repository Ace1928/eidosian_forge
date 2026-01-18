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
def prepare_input_initializer_tensors(self, initializer_shape, input_shape):
    nodes = [make_node('Add', ['x', 'y'], 'z')]
    if initializer_shape is None:
        initializer = []
    else:
        size = 1
        for d in initializer_shape:
            size = size * d
        vals = [0.0 for i in range(size)]
        initializer = [make_tensor('x', TensorProto.FLOAT, initializer_shape, vals), make_tensor('y', TensorProto.FLOAT, initializer_shape, vals)]
    if input_shape is None:
        inputs = []
    else:
        inputs = [helper.make_tensor_value_info('x', TensorProto.FLOAT, input_shape), helper.make_tensor_value_info('y', TensorProto.FLOAT, input_shape)]
    graph = helper.make_graph(nodes, 'test', inputs=inputs, outputs=[], initializer=initializer, value_info=[])
    return helper.make_model(graph)