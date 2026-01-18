from __future__ import annotations
import itertools
import os
import pathlib
import tempfile
import unittest
import uuid
from typing import Any
import numpy as np
import parameterized
import onnx
from onnx import ModelProto, TensorProto, checker, helper, shape_inference
from onnx.external_data_helper import (
from onnx.numpy_helper import from_array, to_array
def create_test_model_proto(self) -> ModelProto:
    tensors = self.create_data_tensors([(self.attribute_value, 'attribute_value'), (self.initializer_value, 'input_value')])
    constant_node = onnx.helper.make_node('Constant', inputs=[], outputs=['values'], value=tensors[0])
    inputs = [helper.make_tensor_value_info('input_value', onnx.TensorProto.FLOAT, self.initializer_value.shape)]
    graph = helper.make_graph([constant_node], 'test_graph', inputs=inputs, outputs=[], initializer=[tensors[1]])
    return helper.make_model(graph)