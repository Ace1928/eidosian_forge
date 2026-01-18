import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_gridsample() -> None:
    node = onnx.helper.make_node('GridSample', inputs=['X', 'Grid'], outputs=['Y'], mode='linear', padding_mode='zeros', align_corners=0)
    X = np.array([[[[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0], [12.0, 13.0, 14.0, 15.0]]]], dtype=np.float32)
    Grid = np.array([[[[-1.0, -1.0], [-0.6, -1.0], [-0.2, -1.0], [0.2, -1.0], [0.6, -1.0], [1.0, -1.0]], [[-1.0, -0.6], [-0.6, -0.6], [-0.2, -0.6], [0.2, -0.6], [0.6, -0.6], [1.0, -0.6]], [[-1.0, -0.2], [-0.6, -0.2], [-0.2, -0.2], [0.2, -0.2], [0.6, -0.2], [1.0, -0.2]], [[-1.0, 0.2], [-0.6, 0.2], [-0.2, 0.2], [0.2, 0.2], [0.6, 0.2], [1.0, 0.2]], [[-1.0, 0.6], [-0.6, 0.6], [-0.2, 0.6], [0.2, 0.6], [0.6, 0.6], [1.0, 0.6]], [[-1.0, 1.0], [-0.6, 1.0], [-0.2, 1.0], [0.2, 1.0], [0.6, 1.0], [1.0, 1.0]]]], dtype=np.float32)
    Y = np.array([[[[0.0, 0.15, 0.55, 0.95, 1.35, 0.75], [0.6, 1.5, 2.3, 3.1, 3.9, 2.1], [2.2, 4.7, 5.5, 6.3, 7.1, 3.7], [3.8, 7.9, 8.7, 9.5, 10.3, 5.3], [5.4, 11.1, 11.9, 12.7, 13.5, 6.9], [3.0, 6.15, 6.55, 6.95, 7.35, 3.75]]]], dtype=np.float32)
    expect(node, inputs=[X, Grid], outputs=[Y], name='test_gridsample')