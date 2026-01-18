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
def ll_op(left, right):
    """Perform a LL operation.

    A LL operation multiplies both left and right circuits
    with the dagger of the left circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a LL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LL operation:

    >>> from sympy.physics.quantum.identitysearch import ll_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> ll_op((x, y, z), ())
    ((Y(0), Z(0)), (X(0),))

    >>> ll_op((y, z), (x,))
    ((Z(0),), (Y(0), X(0)))
    """
    if len(left) > 0:
        ll_gate = left[0]
        ll_gate_is_unitary = is_scalar_matrix((Dagger(ll_gate), ll_gate), _get_min_qubits(ll_gate), True)
    if len(left) > 0 and ll_gate_is_unitary:
        new_left = left[1:len(left)]
        new_right = (Dagger(ll_gate),) + right
        return (new_left, new_right)
    return None