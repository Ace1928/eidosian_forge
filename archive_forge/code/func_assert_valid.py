from string import ascii_lowercase
import copy
import pickle
import numpy as np
import pennylane as qml
from pennylane.operation import EigvalsUndefinedError
def assert_valid(op: qml.operation.Operator, skip_pickle=False) -> None:
    """Runs basic validation checks on an :class:`~.operation.Operator` to make
    sure it has been correctly defined.

    Args:
        op (.Operator): an operator instance to validate

    Keyword Args:
        skip_pickle=False : If ``True``, pickling tests are not run. Set to ``True`` when
            testing a locally defined operator, as pickle cannot handle local objects

    **Examples:**

    .. code-block:: python

        class MyOp(qml.operation.Operator):

            def __init__(self, data, wires):
                self.data = data
                super().__init__(wires=wires)

        op = MyOp(qml.numpy.array(0.5), wires=0)

    .. code-block::

        >>> assert_valid(op)
        AssertionError: op.data must be a tuple

    .. code-block:: python

        class MyOp(qml.operation.Operator):

            def __init__(self, wires):
                self.hyperparameters["unhashable_list"] = []
                super().__init__(wires=wires)

        op = MyOp(wires = 0)
        assert_valid(op)

    .. code-block::

        ValueError: metadata output from _flatten must be hashable. This also applies to hyperparameters

    """
    assert isinstance(op.data, tuple), 'op.data must be a tuple'
    assert isinstance(op.parameters, list), 'op.parameters must be a list'
    for d, p in zip(op.data, op.parameters):
        assert isinstance(d, qml.typing.TensorLike), 'each data element must be tensorlike'
        assert qml.math.allclose(d, p), 'data and parameters must match.'
    if len(op.wires) <= 26:
        _check_wires(op)
    _check_copy(op)
    _check_pytree(op)
    if not skip_pickle:
        _check_pickle(op)
    _check_bind_new_parameters(op)
    _check_decomposition(op)
    _check_matrix(op)
    _check_matrix_matches_decomp(op)
    _check_eigendecomposition(op)