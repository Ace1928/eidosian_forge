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
def get_basis(expr, *, basis=None, replace_none=True, **options):
    """
    Returns a basis state instance corresponding to the basis specified in
    options=s. If no basis is specified, the function tries to form a default
    basis state of the given expression.

    There are three behaviors:

    1. The basis specified in options is already an instance of StateBase. If
       this is the case, it is simply returned. If the class is specified but
       not an instance, a default instance is returned.

    2. The basis specified is an operator or set of operators. If this
       is the case, the operator_to_state mapping method is used.

    3. No basis is specified. If expr is a state, then a default instance of
       its class is returned.  If expr is an operator, then it is mapped to the
       corresponding state.  If it is neither, then we cannot obtain the basis
       state.

    If the basis cannot be mapped, then it is not changed.

    This will be called from within represent, and represent will
    only pass QExpr's.

    TODO (?): Support for Muls and other types of expressions?

    Parameters
    ==========

    expr : Operator or StateBase
        Expression whose basis is sought

    Examples
    ========

    >>> from sympy.physics.quantum.represent import get_basis
    >>> from sympy.physics.quantum.cartesian import XOp, XKet, PxOp, PxKet
    >>> x = XKet()
    >>> X = XOp()
    >>> get_basis(x)
    |x>
    >>> get_basis(X)
    |x>
    >>> get_basis(x, basis=PxOp())
    |px>
    >>> get_basis(x, basis=PxKet)
    |px>

    """
    if basis is None and (not replace_none):
        return None
    if basis is None:
        if isinstance(expr, KetBase):
            return _make_default(expr.__class__)
        elif isinstance(expr, BraBase):
            return _make_default(expr.dual_class())
        elif isinstance(expr, Operator):
            state_inst = operators_to_state(expr)
            return state_inst if state_inst is not None else None
        else:
            return None
    elif isinstance(basis, Operator) or (not isinstance(basis, StateBase) and issubclass(basis, Operator)):
        state = operators_to_state(basis)
        if state is None:
            return None
        elif isinstance(state, StateBase):
            return state
        else:
            return _make_default(state)
    elif isinstance(basis, StateBase):
        return basis
    elif issubclass(basis, StateBase):
        return _make_default(basis)
    else:
        return None