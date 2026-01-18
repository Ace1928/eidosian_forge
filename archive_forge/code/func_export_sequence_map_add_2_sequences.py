import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_sequence_map_add_2_sequences():
    body = onnx.helper.make_graph([onnx.helper.make_node('Add', ['in0', 'in1'], ['out0'])], 'seq_map_body', [onnx.helper.make_tensor_value_info('in0', onnx.TensorProto.FLOAT, ['N']), onnx.helper.make_tensor_value_info('in1', onnx.TensorProto.FLOAT, ['N'])], [onnx.helper.make_tensor_value_info('out0', onnx.TensorProto.FLOAT, ['N'])])
    node = onnx.helper.make_node('SequenceMap', inputs=['x0', 'x1'], outputs=['y0'], body=body)
    N = [np.random.randint(1, 10) for _ in range(3)]
    x0 = [np.random.uniform(0.0, 1.0, N[k]).astype(np.float32) for k in range(3)]
    x1 = [np.random.uniform(0.0, 1.0, N[k]).astype(np.float32) for k in range(3)]
    y0 = [x0[k] + x1[k] for k in range(3)]
    input_type_protos = [onnx.helper.make_sequence_type_proto(onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N'])), onnx.helper.make_sequence_type_proto(onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N']))]
    output_type_protos = [onnx.helper.make_sequence_type_proto(onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['N']))]
    expect(node, inputs=[x0, x1], outputs=[y0], input_type_protos=input_type_protos, output_type_protos=output_type_protos, name='test_sequence_map_add_2_sequences')