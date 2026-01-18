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
def rep_expectation(expr, **options):
    """
    Returns an ``<x'|A|x>`` type representation for the given operator.

    Parameters
    ==========

    expr : Operator
        Operator to be represented in the specified basis

    Examples
    ========

    >>> from sympy.physics.quantum.cartesian import XOp, PxOp, PxKet
    >>> from sympy.physics.quantum.represent import rep_expectation
    >>> rep_expectation(XOp())
    x_1*DiracDelta(x_1 - x_2)
    >>> rep_expectation(XOp(), basis=PxOp())
    <px_2|*X*|px_1>
    >>> rep_expectation(XOp(), basis=PxKet())
    <px_2|*X*|px_1>

    """
    if 'index' not in options:
        options['index'] = 1
    if not isinstance(expr, Operator):
        raise TypeError('The passed expression is not an operator')
    basis_state = get_basis(expr, **options)
    if basis_state is None or not isinstance(basis_state, StateBase):
        raise NotImplementedError('Could not get basis kets for this operator')
    basis_kets = enumerate_states(basis_state, options['index'], 2)
    bra = basis_kets[1].dual
    ket = basis_kets[0]
    return qapply(bra * expr * ket)