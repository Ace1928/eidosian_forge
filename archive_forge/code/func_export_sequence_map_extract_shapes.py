import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_sequence_map_extract_shapes():
    body = onnx.helper.make_graph([onnx.helper.make_node('Shape', ['x'], ['shape'])], 'seq_map_body', [onnx.helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, ['H', 'W', 'C'])], [onnx.helper.make_tensor_value_info('shape', onnx.TensorProto.INT64, [3])])
    node = onnx.helper.make_node('SequenceMap', inputs=['in_seq'], outputs=['shapes'], body=body)
    shapes = [np.array([40, 30, 3], dtype=np.int64), np.array([20, 10, 3], dtype=np.int64), np.array([10, 5, 3], dtype=np.int64)]
    x0 = [np.zeros(shape, dtype=np.float32) for shape in shapes]
    input_type_protos = [onnx.helper.make_sequence_type_proto(onnx.helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ['H', 'W', 'C']))]
    output_type_protos = [onnx.helper.make_sequence_type_proto(onnx.helper.make_tensor_type_proto(onnx.TensorProto.INT64, [3]))]
    expect(node, inputs=[x0], outputs=[shapes], input_type_protos=input_type_protos, output_type_protos=output_type_protos, name='test_sequence_map_extract_shapes')