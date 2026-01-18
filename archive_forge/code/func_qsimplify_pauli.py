from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.physics.quantum import Operator, Ket, Bra
from sympy.physics.quantum import ComplexSpace
from sympy.matrices import Matrix
from sympy.functions.special.tensor_functions import KroneckerDelta
def qsimplify_pauli(e):
    """
    Simplify an expression that includes products of pauli operators.

    Parameters
    ==========

    e : expression
        An expression that contains products of Pauli operators that is
        to be simplified.

    Examples
    ========

    >>> from sympy.physics.quantum.pauli import SigmaX, SigmaY
    >>> from sympy.physics.quantum.pauli import qsimplify_pauli
    >>> sx, sy = SigmaX(), SigmaY()
    >>> sx * sy
    SigmaX()*SigmaY()
    >>> qsimplify_pauli(sx * sy)
    I*SigmaZ()
    """
    if isinstance(e, Operator):
        return e
    if isinstance(e, (Add, Pow, exp)):
        t = type(e)
        return t(*(qsimplify_pauli(arg) for arg in e.args))
    if isinstance(e, Mul):
        c, nc = e.args_cnc()
        nc_s = []
        while nc:
            curr = nc.pop(0)
            while len(nc) and isinstance(curr, SigmaOpBase) and isinstance(nc[0], SigmaOpBase) and (curr.name == nc[0].name):
                x = nc.pop(0)
                y = _qsimplify_pauli_product(curr, x)
                c1, nc1 = y.args_cnc()
                curr = Mul(*nc1)
                c = c + c1
            nc_s.append(curr)
        return Mul(*c) * Mul(*nc_s)
    return e