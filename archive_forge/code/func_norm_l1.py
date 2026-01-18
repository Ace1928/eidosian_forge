import numpy as np
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
@staticmethod
def norm_l1(x):
    """L1 normalization"""
    div = np.abs(x).sum(axis=1).reshape((x.shape[0], -1))
    return x / np.maximum(div, 1e-30)