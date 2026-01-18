import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_if_seq() -> None:
    then_out = onnx.helper.make_tensor_sequence_value_info('then_out', onnx.TensorProto.FLOAT, shape=[5])
    else_out = onnx.helper.make_tensor_sequence_value_info('else_out', onnx.TensorProto.FLOAT, shape=[5])
    x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
    y = [np.array([5, 4, 3, 2, 1]).astype(np.float32)]
    then_const_node = onnx.helper.make_node('Constant', inputs=[], outputs=['x'], value=onnx.numpy_helper.from_array(x[0]))
    then_seq_node = onnx.helper.make_node('SequenceConstruct', inputs=['x'], outputs=['then_out'])
    else_const_node = onnx.helper.make_node('Constant', inputs=[], outputs=['y'], value=onnx.numpy_helper.from_array(y[0]))
    else_seq_node = onnx.helper.make_node('SequenceConstruct', inputs=['y'], outputs=['else_out'])
    then_body = onnx.helper.make_graph([then_const_node, then_seq_node], 'then_body', [], [then_out])
    else_body = onnx.helper.make_graph([else_const_node, else_seq_node], 'else_body', [], [else_out])
    if_node = onnx.helper.make_node('If', inputs=['cond'], outputs=['res'], then_branch=then_body, else_branch=else_body)
    cond = np.array(1).astype(bool)
    res = x if cond else y
    expect(if_node, inputs=[cond], outputs=[res], name='test_if_seq', opset_imports=[onnx.helper.make_opsetid('', 13)])