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
def test_melweightmatrix_with_output_datatype(self):
    graph = self._make_graph([], [make_node('Constant', [], ['num_mel_bins'], value=make_tensor('num_mel_bins', TensorProto.INT64, (), (10,))), make_node('Constant', [], ['dft_length'], value=make_tensor('dft_length', TensorProto.INT64, (), (128,))), make_node('Constant', [], ['sample_rate'], value=make_tensor('sample_rate', TensorProto.INT64, (), (10,))), make_node('Constant', [], ['lower_edge_hertz'], value=make_tensor('lower_edge_hertz', TensorProto.FLOAT, (), (10.0,))), make_node('Constant', [], ['upper_edge_hertz'], value=make_tensor('upper_edge_hertz', TensorProto.FLOAT, (), (100.0,))), make_node('MelWeightMatrix', ['num_mel_bins', 'dft_length', 'sample_rate', 'lower_edge_hertz', 'upper_edge_hertz'], ['output'], output_datatype=TensorProto.DOUBLE)], [])
    self._assert_inferred(graph, [make_tensor_value_info('num_mel_bins', TensorProto.INT64, ()), make_tensor_value_info('dft_length', TensorProto.INT64, ()), make_tensor_value_info('sample_rate', TensorProto.INT64, ()), make_tensor_value_info('lower_edge_hertz', TensorProto.FLOAT, ()), make_tensor_value_info('upper_edge_hertz', TensorProto.FLOAT, ()), make_tensor_value_info('output', TensorProto.DOUBLE, (65, 10))])