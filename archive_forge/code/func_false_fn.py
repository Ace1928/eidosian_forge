from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.ops import cond as tf_cond
def false_fn():
    false_val.append(if_false())
    if true_val and false_val:
        control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return false_val[0]