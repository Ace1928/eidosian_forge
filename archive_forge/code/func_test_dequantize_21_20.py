import contextlib
import struct
import unittest
from typing import Optional, Tuple
import numpy as np
import parameterized
import onnx.version_converter
from onnx import (
@parameterized.parameterized.expand([('per_tensor', (16, 3), (1,), None, None, True), ('per_axis_none_block_shape', (16, 3), (16,), 1, None, True), ('per_axis_zero_block_shape', (16, 3), (16,), 1, 0, True), ('per_tensor_positive_block_shape', (16, 3), (1,), 1, 2, False), ('per_axis_positive_block_shape', (16, 3), (16,), 1, 2, False), ('blocked_2d', (16, 3), (4, 3), 0, 4, False), ('blocked_3d', (4, 3, 32), (4, 3, 8), 2, 4, False)])
def test_dequantize_21_20(self, _: str, y_shape: Tuple[int, ...], scale_shape: Tuple[int, ...], axis: int, block_size: int, compatible: bool) -> None:

    def test(input_shape, scale_shape, axis, block_size) -> None:
        nodes = [helper.make_node('DequantizeLinear', ['X', 'S', 'ZP'], ['Y'], axis=axis, block_size=block_size)]
        graph = helper.make_graph(nodes, 'test', [helper.make_tensor_value_info('X', TensorProto.INT8, input_shape), helper.make_tensor_value_info('S', TensorProto.FLOAT, scale_shape), helper.make_tensor_value_info('ZP', TensorProto.INT8, scale_shape)], [helper.make_tensor_value_info('Y', TensorProto.FLOAT, input_shape)])
        _ = self._converted(graph, helper.make_operatorsetid('', 21), 20)
    context_manager = contextlib.nullcontext() if compatible else self.assertRaises(RuntimeError)
    with context_manager:
        test(y_shape, scale_shape, axis, block_size)