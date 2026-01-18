import unittest
import automatic_conversion_test_base
import numpy as np
import onnx
from onnx import TensorProto, helper
def test_If(self) -> None:
    sub_output = [helper.make_tensor_value_info('out', TensorProto.FLOAT, [3, 4, 5])]
    then_tensor = helper.make_tensor('Value', TensorProto.FLOAT, dims=[3, 4, 5], vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True)
    then_node = helper.make_node('Constant', [], ['out'], value=then_tensor)
    then_graph = helper.make_graph([then_node], 'then_graph', [], sub_output, [])
    else_tensor = helper.make_tensor('Value', TensorProto.FLOAT, dims=[3, 4, 5], vals=np.random.rand(3, 4, 5).astype(np.float32).tobytes(), raw=True)
    else_node = helper.make_node('Constant', [], ['out'], value=else_tensor)
    else_graph = helper.make_graph([else_node], 'else_graph', [], sub_output, [])
    self._test_op_upgrade('If', 1, [[0]], [[3, 4, 5]], [TensorProto.BOOL], attrs={'then_branch': then_graph, 'else_branch': else_graph})