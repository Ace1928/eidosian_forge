import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_scan_8() -> None:
    sum_in = onnx.helper.make_tensor_value_info('sum_in', onnx.TensorProto.FLOAT, [2])
    next = onnx.helper.make_tensor_value_info('next', onnx.TensorProto.FLOAT, [2])
    sum_out = onnx.helper.make_tensor_value_info('sum_out', onnx.TensorProto.FLOAT, [2])
    scan_out = onnx.helper.make_tensor_value_info('scan_out', onnx.TensorProto.FLOAT, [2])
    add_node = onnx.helper.make_node('Add', inputs=['sum_in', 'next'], outputs=['sum_out'])
    id_node = onnx.helper.make_node('Identity', inputs=['sum_out'], outputs=['scan_out'])
    scan_body = onnx.helper.make_graph([add_node, id_node], 'scan_body', [sum_in, next], [sum_out, scan_out])
    no_sequence_lens = ''
    node = onnx.helper.make_node('Scan', inputs=[no_sequence_lens, 'initial', 'x'], outputs=['y', 'z'], num_scan_inputs=1, body=scan_body)
    initial = np.array([0, 0]).astype(np.float32).reshape((1, 2))
    x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape((1, 3, 2))
    y = np.array([9, 12]).astype(np.float32).reshape((1, 2))
    z = np.array([1, 2, 4, 6, 9, 12]).astype(np.float32).reshape((1, 3, 2))
    expect(node, inputs=[initial, x], outputs=[y, z], name='test_scan_sum', opset_imports=[onnx.helper.make_opsetid('', 8)])