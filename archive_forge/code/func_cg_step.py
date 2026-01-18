import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export
def cg_step(i, state):
    z = math_ops.matvec(operator, state.p)
    alpha = state.gamma / dot(state.p, z)
    x = state.x + alpha[..., array_ops.newaxis] * state.p
    r = state.r - alpha[..., array_ops.newaxis] * z
    if preconditioner is None:
        q = r
    else:
        q = preconditioner.matvec(r)
    gamma = dot(r, q)
    beta = gamma / state.gamma
    p = q + beta[..., array_ops.newaxis] * state.p
    return (i + 1, cg_state(i + 1, x, r, p, gamma))