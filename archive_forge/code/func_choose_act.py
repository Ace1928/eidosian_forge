import numpy as np
from onnx.reference.op_run import OpRun
def choose_act(self, name, alpha, beta):
    if name in ('Tanh', 'tanh'):
        return self._f_tanh
    if name in ('Affine', 'affine'):
        return lambda x: x * alpha + beta
    raise RuntimeError(f'Unknown activation function {name!r}.')