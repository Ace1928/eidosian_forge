import numpy as np
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
@staticmethod
def norm_max(x):
    """Max normalization"""
    div = np.abs(x).max(axis=1).reshape((x.shape[0], -1))
    return x / np.maximum(div, 1e-30)