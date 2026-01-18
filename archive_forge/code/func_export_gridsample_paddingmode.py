import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_gridsample_paddingmode() -> None:
    X = np.array([[[[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]]], dtype=np.float32)
    Grid = np.array([[[[-10.0, -10.0], [-5.0, -5.0], [-0.2, -0.2], [10.0, 10.0]], [[10.0, 10.0], [-0.2, -0.2], [5.0, 5.0], [10.0, 10.0]]]], dtype=np.float32)
    node = onnx.helper.make_node('GridSample', inputs=['X', 'Grid'], outputs=['Y'], padding_mode='zeros')
    Y_zeros = np.array([[[[0.0, 0.0, 1.7, 0.0], [0.0, 1.7, 0.0, 0.0]]]], dtype=np.float32)
    expect(node, inputs=[X, Grid], outputs=[Y_zeros], name='test_gridsample_zeros_padding')
    node = onnx.helper.make_node('GridSample', inputs=['X', 'Grid'], outputs=['Y'], padding_mode='border')
    Y_border = np.array([[[[0.0, 0.0, 1.7, 5.0], [5.0, 1.7, 5.0, 5.0]]]], dtype=np.float32)
    expect(node, inputs=[X, Grid], outputs=[Y_border], name='test_gridsample_border_padding')
    node = onnx.helper.make_node('GridSample', inputs=['X', 'Grid'], outputs=['Y'], padding_mode='reflection')
    Y_reflection = np.array([[[[2.5, 0.0, 1.7, 2.5], [2.5, 1.7, 5.0, 2.5]]]], dtype=np.float32)
    expect(node, inputs=[X, Grid], outputs=[Y_reflection], name='test_gridsample_reflection_padding')