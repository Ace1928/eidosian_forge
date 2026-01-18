from typing import Any, List
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_loop_16_none() -> None:
    ten_in_tp = onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, [])
    seq_in_tp = onnx.helper.make_sequence_type_proto(ten_in_tp)
    opt_in_tp = onnx.helper.make_optional_type_proto(seq_in_tp)
    opt_in = onnx.helper.make_value_info('opt_seq_in', opt_in_tp)
    seq_out = onnx.helper.make_tensor_sequence_value_info('seq_out', onnx.TensorProto.FLOAT, [])
    cond_in = onnx.helper.make_tensor_value_info('cond_in', onnx.TensorProto.BOOL, [])
    cond_out = onnx.helper.make_tensor_value_info('cond_out', onnx.TensorProto.BOOL, [])
    iter_count = onnx.helper.make_tensor_value_info('iter_count', onnx.TensorProto.INT64, [])
    x0 = np.array(0).astype(np.float32)
    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)
    optional_has_elem_node = onnx.helper.make_node('OptionalHasElement', inputs=['opt_seq_in'], outputs=['optional_has_elem'])
    optional_is_none = onnx.helper.make_node('Not', inputs=['optional_has_elem'], outputs=['optional_is_none'])
    optional_get_elem = onnx.helper.make_node('OptionalGetElement', inputs=['opt_seq_in'], outputs=['seq_in'])
    constant_in = onnx.helper.make_node('Constant', inputs=[], outputs=['constant_in'], value=onnx.helper.make_tensor(name='const_tensor', data_type=onnx.TensorProto.FLOAT, dims=(), vals=[0]))
    seq_const_in = onnx.helper.make_node('SequenceConstruct', inputs=['constant_in'], outputs=['init_seq_in'])
    then_seq_out = onnx.helper.make_tensor_sequence_value_info('init_seq_in', onnx.TensorProto.FLOAT, [])
    then_body = onnx.helper.make_graph([constant_in, seq_const_in], 'then_body', [], [then_seq_out])
    else_seq_out = onnx.helper.make_tensor_sequence_value_info('seq_in', onnx.TensorProto.FLOAT, [])
    else_body = onnx.helper.make_graph([optional_get_elem], 'else_body', [], [else_seq_out])
    if_node = onnx.helper.make_node('If', inputs=['optional_is_none'], outputs=['sequence'], then_branch=then_body, else_branch=else_body)
    x_const_node = onnx.helper.make_node('Constant', inputs=[], outputs=['x'], value=onnx.helper.make_tensor(name='const_tensor_x', data_type=onnx.TensorProto.FLOAT, dims=x.shape, vals=x.flatten().astype(float)))
    one_const_node = onnx.helper.make_node('Constant', inputs=[], outputs=['one'], value=onnx.helper.make_tensor(name='const_tensor_one', data_type=onnx.TensorProto.INT64, dims=(), vals=[1]))
    zero_const_node = onnx.helper.make_node('Constant', inputs=[], outputs=['slice_start'], value=onnx.helper.make_tensor(name='const_tensor_zero', data_type=onnx.TensorProto.INT64, dims=(1,), vals=[0]))
    axes_node = onnx.helper.make_node('Constant', inputs=[], outputs=['axes'], value=onnx.helper.make_tensor(name='const_tensor_axes', data_type=onnx.TensorProto.INT64, dims=(), vals=[0]))
    add_node = onnx.helper.make_node('Add', inputs=['iter_count', 'one'], outputs=['end'])
    end_unsqueeze_node = onnx.helper.make_node('Unsqueeze', inputs=['end', 'axes'], outputs=['slice_end'])
    slice_node = onnx.helper.make_node('Slice', inputs=['x', 'slice_start', 'slice_end'], outputs=['slice_out'])
    insert_node = onnx.helper.make_node('SequenceInsert', inputs=['sequence', 'slice_out'], outputs=['seq_out'])
    identity_node = onnx.helper.make_node('Identity', inputs=['cond_in'], outputs=['cond_out'])
    loop_body = onnx.helper.make_graph([identity_node, optional_has_elem_node, optional_is_none, if_node, x_const_node, one_const_node, zero_const_node, add_node, axes_node, end_unsqueeze_node, slice_node, insert_node], 'loop_body', [iter_count, cond_in, opt_in], [cond_out, seq_out])
    node = onnx.helper.make_node('Loop', inputs=['trip_count', 'cond', 'opt_seq'], outputs=['seq_res'], body=loop_body)
    trip_count = np.array(5).astype(np.int64)
    cond = np.array(1).astype(bool)
    seq_res = compute_loop_outputs(x, [x0], trip_count)
    opt_seq_in: List[Any] = [x0]
    expect(node, inputs=[trip_count, cond, opt_seq_in], outputs=[seq_res], name='test_loop16_seq_none', opset_imports=[onnx.helper.make_opsetid('', 16)], input_type_protos=[onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, trip_count.shape), onnx.helper.make_tensor_type_proto(onnx.TensorProto.BOOL, cond.shape), opt_in_tp])