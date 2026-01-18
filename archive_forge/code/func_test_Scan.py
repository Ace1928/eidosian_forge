import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Scan(self) -> None:
    sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
    next_in = onnx.helper.make_tensor_value_info('next_in', onnx.TensorProto.FLOAT, [2])
    sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
    scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])
    add_node = onnx.helper.make_node('Add', inputs=['sum_in', 'next_in'], outputs=['sum_out'])
    id_node = onnx.helper.make_node('Identity', inputs=['sum_out'], outputs=['scan_out'])
    body = onnx.helper.make_graph([add_node, id_node], 'scan_body', [sum_in, next_in], [sum_out, scan_out])
    self._test_op_upgrade('Scan', 8, ['', [1, 2], [1, 3, 2]], [[1, 2], [1, 3, 2]], attrs={'body': body, 'num_scan_inputs': 1})