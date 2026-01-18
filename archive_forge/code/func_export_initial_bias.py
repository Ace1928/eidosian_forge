from typing import Any, Tuple
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_initial_bias() -> None:
    input = np.array([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]).astype(np.float32)
    input_size = 3
    hidden_size = 5
    custom_bias = 0.1
    weight_scale = 0.1
    node = onnx.helper.make_node('RNN', inputs=['X', 'W', 'R', 'B'], outputs=['', 'Y_h'], hidden_size=hidden_size)
    W = weight_scale * np.ones((1, hidden_size, input_size)).astype(np.float32)
    R = weight_scale * np.ones((1, hidden_size, hidden_size)).astype(np.float32)
    W_B = custom_bias * np.ones((1, hidden_size)).astype(np.float32)
    R_B = np.zeros((1, hidden_size)).astype(np.float32)
    B = np.concatenate((W_B, R_B), axis=1)
    rnn = RNNHelper(X=input, W=W, R=R, B=B)
    _, Y_h = rnn.step()
    expect(node, inputs=[input, W, R, B], outputs=[Y_h.astype(np.float32)], name='test_simple_rnn_with_initial_bias')