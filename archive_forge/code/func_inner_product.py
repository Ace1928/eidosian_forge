from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
def inner_product(u, v):
    u = _to_matrix(u)
    v = _to_matrix(v)
    return math_ops.matmul(u, v, transpose_b=True)