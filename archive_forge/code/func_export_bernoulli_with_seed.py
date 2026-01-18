import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_bernoulli_with_seed() -> None:
    seed = float(0)
    node = onnx.helper.make_node('Bernoulli', inputs=['x'], outputs=['y'], seed=seed)
    x = np.random.uniform(0.0, 1.0, 10).astype(np.float32)
    y = bernoulli_reference_implementation(x, np.float32)
    expect(node, inputs=[x], outputs=[y], name='test_bernoulli_seed')