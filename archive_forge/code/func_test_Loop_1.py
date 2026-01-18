import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_Loop_1(self) -> None:
    iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])
    cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
    x_in = onnx.helper.make_tensor_value_info('x_in', onnx.TensorProto.FLOAT, [1])
    cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
    x_out = onnx.helper.make_tensor_value_info('x_out', onnx.TensorProto.FLOAT, [1])
    x_scan = onnx.helper.make_tensor_value_info('x_scan', onnx.TensorProto.FLOAT, [1])
    const = onnx.helper.make_node('Constant', inputs=[], outputs=['one'], value=onnx.helper.make_tensor(name='value', data_type=onnx.TensorProto.FLOAT, dims=[1], vals=np.array([1]).astype(np.float32).astype(float)))
    add = onnx.helper.make_node('Add', inputs=['x_in', 'one'], outputs=['x_out'])
    id_1 = onnx.helper.make_node('Identity', inputs=['x_out'], outputs=['x_scan'])
    id_2 = onnx.helper.make_node('Identity', inputs=['cond_in'], outputs=['cond_out'])
    loop_body = onnx.helper.make_graph([const, add, id_1, id_2], 'loop_body', [iter_count, cond_in, x_in], [cond_out, x_out, x_scan])
    self._test_op_upgrade('Loop', 1, [[], '', [1]], [[1], [5, 1]], [TensorProto.INT64, TensorProto.BOOL, TensorProto.FLOAT], attrs={'body': loop_body})