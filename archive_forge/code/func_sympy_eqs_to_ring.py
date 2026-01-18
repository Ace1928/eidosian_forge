from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import connected_components
from sympy.core.sympify import sympify
from sympy.core.numbers import Integer, Rational
from sympy.matrices.dense import MutableDenseMatrix
from sympy.polys.domains import ZZ, QQ
from sympy.polys.domains import EX
from sympy.polys.rings import sring
from sympy.polys.polyerrors import NotInvertible
from sympy.polys.domainmatrix import DomainMatrix
def sympy_eqs_to_ring(eqs, symbols):
    """Convert a system of equations from Expr to a PolyRing

    Explanation
    ===========

    High-level functions like ``solve`` expect Expr as inputs but can use
    ``solve_lin_sys`` internally. This function converts equations from
    ``Expr`` to the low-level poly types used by the ``solve_lin_sys``
    function.

    Parameters
    ==========

    eqs: List of Expr
        A list of equations as Expr instances
    symbols: List of Symbol
        A list of the symbols that are the unknowns in the system of
        equations.

    Returns
    =======

    Tuple[List[PolyElement], Ring]: The equations as PolyElement instances
    and the ring of polynomials within which each equation is represented.

    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.polys.solvers import sympy_eqs_to_ring
    >>> a, x, y = symbols('a, x, y')
    >>> eqs = [x-y, x+a*y]
    >>> eqs_ring, ring = sympy_eqs_to_ring(eqs, [x, y])
    >>> eqs_ring
    [x - y, x + a*y]
    >>> type(eqs_ring[0])
    <class 'sympy.polys.rings.PolyElement'>
    >>> ring
    ZZ(a)[x,y]

    With the equations in this form they can be passed to ``solve_lin_sys``:

    >>> from sympy.polys.solvers import solve_lin_sys
    >>> solve_lin_sys(eqs_ring, ring)
    {y: 0, x: 0}
    """
    try:
        K, eqs_K = sring(eqs, symbols, field=True, extension=True)
    except NotInvertible:
        K, eqs_K = sring(eqs, symbols, domain=EX)
    return (eqs_K, K.to_domain())