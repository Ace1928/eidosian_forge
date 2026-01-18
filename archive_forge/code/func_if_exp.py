from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.ops import cond as tf_cond
def if_exp(cond, if_true, if_false, expr_repr):
    if tensors.is_dense_tensor(cond):
        return _tf_if_exp(cond, if_true, if_false, expr_repr)
    else:
        return _py_if_exp(cond, if_true, if_false)