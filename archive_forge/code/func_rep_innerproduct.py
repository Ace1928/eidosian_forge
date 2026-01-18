from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.integrals.integrals import integrate
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.innerproduct import InnerProduct
from sympy.physics.quantum.qexpr import QExpr
from sympy.physics.quantum.tensorproduct import TensorProduct
from sympy.physics.quantum.matrixutils import flatten_scalar
from sympy.physics.quantum.state import KetBase, BraBase, StateBase
from sympy.physics.quantum.operator import Operator, OuterProduct
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.operatorset import operators_to_state, state_to_operators
def rep_innerproduct(expr, **options):
    """
    Returns an innerproduct like representation (e.g. ``<x'|x>``) for the
    given state.

    Attempts to calculate inner product with a bra from the specified
    basis. Should only be passed an instance of KetBase or BraBase

    Parameters
    ==========

    expr : KetBase or BraBase
        The expression to be represented

    Examples
    ========

    >>> from sympy.physics.quantum.represent import rep_innerproduct
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> rep_innerproduct(XKet())
    DiracDelta(x - x_1)
    >>> rep_innerproduct(XKet(), basis=PxOp())
    sqrt(2)*exp(-I*px_1*x/hbar)/(2*sqrt(hbar)*sqrt(pi))
    >>> rep_innerproduct(PxKet(), basis=XOp())
    sqrt(2)*exp(I*px*x_1/hbar)/(2*sqrt(hbar)*sqrt(pi))

    """
    if not isinstance(expr, (KetBase, BraBase)):
        raise TypeError('expr passed is not a Bra or Ket')
    basis = get_basis(expr, **options)
    if not isinstance(basis, StateBase):
        raise NotImplementedError("Can't form this representation!")
    if 'index' not in options:
        options['index'] = 1
    basis_kets = enumerate_states(basis, options['index'], 2)
    if isinstance(expr, BraBase):
        bra = expr
        ket = basis_kets[1] if basis_kets[0].dual == expr else basis_kets[0]
    else:
        bra = basis_kets[1].dual if basis_kets[0] == expr else basis_kets[0].dual
        ket = expr
    prod = InnerProduct(bra, ket)
    result = prod.doit()
    format = options.get('format', 'sympy')
    return expr._format_represent(result, format)