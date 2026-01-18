from typing import Tuple
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_einsum_inner_prod() -> None:
    Eqn = 'i,i'
    node = onnx.helper.make_node('Einsum', inputs=['x', 'y'], outputs=['z'], equation=Eqn)
    X = np.random.randn(5)
    Y = np.random.randn(5)
    Z = einsum_reference_implementation(Eqn, (X, Y))
    expect(node, inputs=[X, Y], outputs=[Z], name='test_einsum_inner_prod')