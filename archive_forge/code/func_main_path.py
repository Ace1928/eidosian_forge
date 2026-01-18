from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import take_while_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
def main_path():
    body(iterate)
    new_loop_vars = get_state()
    control_flow.verify_tf_loop_vars(init_vars, loop_vars, new_loop_vars, symbol_names, opts, check_shapes=False)
    return new_loop_vars