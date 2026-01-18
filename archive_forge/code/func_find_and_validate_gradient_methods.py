from functools import partial
import warnings
import pennylane as qml
from pennylane.transforms.tape_expand import expand_invalid_trainable
from pennylane.measurements import (
def find_and_validate_gradient_methods(tape, method, trainable_param_indices, use_graph=True):
    """Returns a dictionary of gradient methods for each trainable parameter after
    validating if the gradient method requested is supported by the trainable parameters

    Parameter gradient methods include:

    * ``None``: the parameter does not support differentiation.

    * ``"0"``: the variational circuit output does not depend on this
      parameter (the partial derivative is zero).

    In addition, the operator might define its own grad method
    via :attr:`.Operator.grad_method`.

    Args:
        tape (`~.QuantumScript`): the tape to analyze
        method (str): the gradient method to use
        trainable_param_indices (list[int]): the indices of the trainable parameters
            to compute the Jacobian for
        use_graph (bool): whether to use the circuit graph to find if
            a parameter has zero gradient

    Returns:
        dict: dictionary of the gradient methods for each trainable parameter

    Raises:
        ValueError: If there exist non-differentiable trainable parameters on the tape.
        ValueError: If the Jacobian method is ``"analytic"`` but there exist some trainable
            parameters on the tape that only support numeric differentiation.

    """
    diff_methods = _find_gradient_methods(tape, trainable_param_indices, use_graph=use_graph)
    _validate_gradient_methods(tape, method, diff_methods)
    return diff_methods