import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_if_optional() -> None:
    ten_in_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=[5])
    seq_in_tp = onnx.helper.make_sequence_type_proto(ten_in_tp)
    then_out_tensor_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=[5])
    then_out_seq_tp = onnx.helper.make_sequence_type_proto(then_out_tensor_tp)
    then_out_opt_tp = onnx.helper.make_optional_type_proto(then_out_seq_tp)
    then_out = onnx.helper.make_value_info('optional_empty', then_out_opt_tp)
    else_out_tensor_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, shape=[5])
    else_out_seq_tp = onnx.helper.make_sequence_type_proto(else_out_tensor_tp)
    else_out_opt_tp = onnx.helper.make_optional_type_proto(else_out_seq_tp)
    else_out = onnx.helper.make_value_info('else_opt', else_out_opt_tp)
    x = [np.array([1, 2, 3, 4, 5]).astype(np.float32)]
    cond = np.array(0).astype(bool)
    res = compute_if_outputs(x, cond)
    opt_empty_in = onnx.helper.make_node('Optional', inputs=[], outputs=['optional_empty'], type=seq_in_tp)
    then_body = onnx.helper.make_graph([opt_empty_in], 'then_body', [], [then_out])
    else_const_node = onnx.helper.make_node('Constant', inputs=[], outputs=['x'], value=onnx.numpy_helper.from_array(x[0]))
    else_seq_node = onnx.helper.make_node('SequenceConstruct', inputs=['x'], outputs=['else_seq'])
    else_optional_seq_node = onnx.helper.make_node('Optional', inputs=['else_seq'], outputs=['else_opt'])
    else_body = onnx.helper.make_graph([else_const_node, else_seq_node, else_optional_seq_node], 'else_body', [], [else_out])
    if_node = onnx.helper.make_node('If', inputs=['cond'], outputs=['sequence'], then_branch=then_body, else_branch=else_body)
    expect(if_node, inputs=[cond], outputs=[res], name='test_if_opt', output_type_protos=[else_out_opt_tp], opset_imports=[onnx.helper.make_opsetid('', 16)])