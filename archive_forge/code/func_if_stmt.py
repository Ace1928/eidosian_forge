import functools
import sys
import traceback
import numpy as np
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.operators import variables
from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.autograph.utils import misc
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.types import distribute
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
def if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts):
    """Functional form of an if statement.

  The conditional operates on a state, which includes all symbols whose values
  are a function of the branch taken.

  For example, given the code below that calculates the abs function:

  ```
    x = 1
    if x > 0:
      x = -x
  ```

  The state is represented by the variable `x`. The `body, `orelse` and
  `set_state` functions must bind to the original `x` symbol, using `nonlocal`.

  The inputs and outputs of the callables representing the loop blocks are not
  explicit - instead, these functions must use nonlocal/global for side effects.
  The inputs and outputs are instead controlled by the set_state/get_state
  functions.

  Args:
    cond: Boolean.
    body: Callable representing the main block of the conditional.
    orelse: Callable representing the else block of the conditional.
    get_state: Function that returns a tuple containing the values of all
      composite symbols modified within the conditional. This allows access to
      state that branches may mutate through side effects. This function is not
      needed and should not be called when dispatching to code matching Python's
      default semantics. This is useful for checkpointing to avoid unintended
      side-effects when staging requires evaluating all code-paths.
    set_state: Function to set the values of all composite symbols modified
      within the conditional. This is the complement to get_state, used to
      restore checkpointed values. The single argument a tuple containing values
      for each composite symbol that may be modified in a branch of the
      conditional. The is usually the result of a call to get_state.
    symbol_names: Tuple containing basic loop var names.
    nouts: Number of variables output by the statement. Vars which are not
      outputs will not be passed through staged control flow such as tf.cond.
      This includes variables that are defined before the conditional, but are
      not used after it.
  """
    if tensors.is_dense_tensor(cond):
        _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
    else:
        _py_if_stmt(cond, body, orelse)