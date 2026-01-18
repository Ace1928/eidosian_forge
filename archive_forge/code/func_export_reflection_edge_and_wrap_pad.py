import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_reflection_edge_and_wrap_pad() -> None:
    for mode in ('edge', 'reflect', 'wrap'):
        node = onnx.helper.make_node('Pad', inputs=['x', 'pads'], outputs=['y'], mode=mode)
        x = np.random.randn(1, 3, 4, 5).astype(np.int32)
        pads = np.array([0, 0, 1, 1, 0, 0, 1, 1]).astype(np.int64)
        y = pad_impl(x, pads, mode)
        expect(node, inputs=[x, pads], outputs=[y], name=f'test_{mode}_pad')