import numpy as np
from onnx.reference.ops.aionnxml._common_classifier import (
from onnx.reference.ops.aionnxml._op_run_aionnxml import OpRunAiOnnxMl
from onnx.reference.ops.aionnxml.op_svm_helper import SVMCommon
def multiclass_probability(k, R):
    max_iter = max(100, k)
    Q = np.empty((k, k), dtype=R.dtype)
    Qp = np.empty((k,), dtype=R.dtype)
    P = np.empty((k,), dtype=R.dtype)
    eps = 0.005 / k
    for t in range(0, k):
        P[t] = 1.0 / k
        Q[t, t] = (R[:t, t] ** 2).sum()
        Q[t, :t] = Q[:t, t]
        Q[t, t] += (R[t + 1:, t] ** 2).sum()
        Q[t, t + 1:] = -R[t + 1:, t] @ R[t, t + 1:]
    for _ in range(max_iter):
        Qp[:] = Q @ P
        pQp = (P * Qp).sum()
        max_error = 0
        for t in range(0, k):
            error = np.abs(Qp[t] - pQp)
            if error > max_error:
                max_error = error
        if max_error < eps:
            break
        for t in range(k):
            diff = (-Qp[t] + pQp) / Q[t, t]
            P[t] += diff
            pQp = (pQp + diff * (diff * Q[t, t] + 2 * Qp[t])) / (1 + diff) ** 2
            P /= 1 + diff
            Qp[:] = (Qp + diff * Q[t, :]) / (1 + diff)
    return P