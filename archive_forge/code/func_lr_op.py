from collections import deque
from sympy.core.random import randint
from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger
def lr_op(left, right):
    """Perform a LR operation.

    A LR operation multiplies both left and right circuits
    with the dagger of the left circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a LR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LR operation:

    >>> from sympy.physics.quantum.identitysearch import lr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> lr_op((x, y, z), ())
    ((X(0), Y(0)), (Z(0),))

    >>> lr_op((x, y), (z,))
    ((X(0),), (Z(0), Y(0)))
    """
    if len(left) > 0:
        lr_gate = left[len(left) - 1]
        lr_gate_is_unitary = is_scalar_matrix((Dagger(lr_gate), lr_gate), _get_min_qubits(lr_gate), True)
    if len(left) > 0 and lr_gate_is_unitary:
        new_left = left[0:len(left) - 1]
        new_right = right + (Dagger(lr_gate),)
        return (new_left, new_right)
    return None