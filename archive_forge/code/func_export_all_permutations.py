import itertools
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_all_permutations() -> None:
    shape = (2, 3, 4)
    data = np.random.random_sample(shape).astype(np.float32)
    permutations = list(itertools.permutations(np.arange(len(shape))))
    for i, permutation in enumerate(permutations):
        node = onnx.helper.make_node('Transpose', inputs=['data'], outputs=['transposed'], perm=permutation)
        transposed = np.transpose(data, permutation)
        expect(node, inputs=[data], outputs=[transposed], name=f'test_transpose_all_permutations_{i}')