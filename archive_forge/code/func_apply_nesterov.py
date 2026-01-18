import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.defs import AI_ONNX_PREVIEW_TRAINING_DOMAIN
def apply_nesterov(r, t, x, g, v, norm_coefficient, alpha, beta):
    g_regularized = norm_coefficient * x + g
    beta_adjusted = beta if t > 0 else 1
    v_new = alpha * v + beta_adjusted * g_regularized
    x_new = x - r * (g_regularized + alpha * v_new)
    return (x_new, v_new)