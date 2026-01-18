import abc
import itertools
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_v2
from tensorflow.python.ops.linalg import linalg_impl as linalg
from tensorflow.python.ops.linalg import linear_operator_util
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load as load_model
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import save as save_model
from tensorflow.python.util import nest
def check_tape_safe(self, operator, skip_options=None):
    """Check gradients are not None w.r.t. operator.variables.

    Meant to be called from the derived class.

    This ensures grads are not w.r.t every variable in operator.variables.  If
    more fine-grained testing is needed, a custom test should be written.

    Args:
      operator: LinearOperator.  Exact checks done will depend on hints.
      skip_options: Optional list of CheckTapeSafeSkipOptions.
        Makes this test skip particular checks.
    """
    skip_options = skip_options or []
    if not operator.variables:
        raise AssertionError('`operator.variables` was empty')

    def _assert_not_none(iterable):
        for item in iterable:
            self.assertIsNotNone(item)
    with backprop.GradientTape() as tape:
        grad = tape.gradient(operator.to_dense(), operator.variables)
        _assert_not_none(grad)
    with backprop.GradientTape() as tape:
        var_grad = tape.gradient(operator, operator.variables)
        _assert_not_none(var_grad)
        nest.assert_same_structure(var_grad, grad)
    with backprop.GradientTape() as tape:
        _assert_not_none(tape.gradient(operator.adjoint().to_dense(), operator.variables))
    x = math_ops.cast(array_ops.ones(shape=operator.H.shape_tensor()[:-1]), operator.dtype)
    with backprop.GradientTape() as tape:
        _assert_not_none(tape.gradient(operator.matvec(x), operator.variables))
    if not operator.is_square:
        return
    for option in [CheckTapeSafeSkipOptions.DETERMINANT, CheckTapeSafeSkipOptions.LOG_ABS_DETERMINANT, CheckTapeSafeSkipOptions.DIAG_PART, CheckTapeSafeSkipOptions.TRACE]:
        with backprop.GradientTape() as tape:
            if option not in skip_options:
                _assert_not_none(tape.gradient(getattr(operator, option)(), operator.variables))
    if operator.is_non_singular is False:
        return
    with backprop.GradientTape() as tape:
        _assert_not_none(tape.gradient(operator.inverse().to_dense(), operator.variables))
    with backprop.GradientTape() as tape:
        _assert_not_none(tape.gradient(operator.solvevec(x), operator.variables))
    if not (operator.is_self_adjoint and operator.is_positive_definite):
        return
    with backprop.GradientTape() as tape:
        _assert_not_none(tape.gradient(operator.cholesky().to_dense(), operator.variables))