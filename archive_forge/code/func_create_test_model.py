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
def create_test_model(self) -> ModelProto:
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, self.large_data.shape)
    input_init = helper.make_tensor(name='X', data_type=TensorProto.FLOAT, dims=self.large_data.shape, vals=self.large_data.tobytes(), raw=True)
    shape_data = np.array(self.small_data, np.int64)
    shape_init = helper.make_tensor(name='Shape', data_type=TensorProto.INT64, dims=shape_data.shape, vals=shape_data.tobytes(), raw=True)
    C = helper.make_tensor_value_info('C', TensorProto.INT64, self.small_data)
    reshape = onnx.helper.make_node('Reshape', inputs=['X', 'Shape'], outputs=['Y'])
    cast = onnx.helper.make_node('Cast', inputs=['Y'], outputs=['C'], to=TensorProto.INT64)
    graph_def = helper.make_graph([reshape, cast], 'test-model', [X], [C], initializer=[input_init, shape_init])
    model = helper.make_model(graph_def, producer_name='onnx-example')
    return model