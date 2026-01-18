from collections import defaultdict
from sympy.core import Basic, Mul, Add, Pow, sympify
from sympy.core.containers import Tuple, OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol
from sympy.matrices import (MatrixBase, Matrix, ImmutableMatrix,
from sympy.matrices.expressions import (MatrixExpr, MatrixSymbol, MatMul,
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.polys.rootoftools import RootOf
from sympy.utilities.iterables import numbered_symbols, sift, \
from . import cse_opts
def opt_cse(exprs, order='canonical'):
    """Find optimization opportunities in Adds, Muls, Pows and negative
    coefficient Muls.

    Parameters
    ==========

    exprs : list of SymPy expressions
        The expressions to optimize.
    order : string, 'none' or 'canonical'
        The order by which Mul and Add arguments are processed. For large
        expressions where speed is a concern, use the setting order='none'.

    Returns
    =======

    opt_subs : dictionary of expression substitutions
        The expression substitutions which can be useful to optimize CSE.

    Examples
    ========

    >>> from sympy.simplify.cse_main import opt_cse
    >>> from sympy.abc import x
    >>> opt_subs = opt_cse([x**-2])
    >>> k, v = list(opt_subs.keys())[0], list(opt_subs.values())[0]
    >>> print((k, v.as_unevaluated_basic()))
    (x**(-2), 1/(x**2))
    """
    opt_subs = {}
    adds = OrderedSet()
    muls = OrderedSet()
    seen_subexp = set()
    collapsible_subexp = set()

    def _find_opts(expr):
        if not isinstance(expr, (Basic, Unevaluated)):
            return
        if expr.is_Atom or expr.is_Order:
            return
        if iterable(expr):
            list(map(_find_opts, expr))
            return
        if expr in seen_subexp:
            return expr
        seen_subexp.add(expr)
        list(map(_find_opts, expr.args))
        if not isinstance(expr, MatrixExpr) and expr.could_extract_minus_sign():
            if isinstance(expr, Add):
                neg_expr = Add(*(-i for i in expr.args))
            else:
                neg_expr = -expr
            if not neg_expr.is_Atom:
                opt_subs[expr] = Unevaluated(Mul, (S.NegativeOne, neg_expr))
                seen_subexp.add(neg_expr)
                expr = neg_expr
        if isinstance(expr, (Mul, MatMul)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                muls.add(expr)
        elif isinstance(expr, (Add, MatAdd)):
            if len(expr.args) == 1:
                collapsible_subexp.add(expr)
            else:
                adds.add(expr)
        elif isinstance(expr, Inverse):
            pass
        elif isinstance(expr, (Pow, MatPow)):
            base, exp = (expr.base, expr.exp)
            if exp.could_extract_minus_sign():
                opt_subs[expr] = Unevaluated(Pow, (Pow(base, -exp), -1))
    for e in exprs:
        if isinstance(e, (Basic, Unevaluated)):
            _find_opts(e)
    edges = [(s, s.args[0]) for s in collapsible_subexp if s.args[0] in collapsible_subexp]
    for e in reversed(topological_sort((collapsible_subexp, edges))):
        opt_subs[e] = opt_subs.get(e.args[0], e.args[0])
    commutative_muls = OrderedSet()
    for m in muls:
        c, nc = m.args_cnc(cset=False)
        if c:
            c_mul = m.func(*c)
            if nc:
                if c_mul == 1:
                    new_obj = m.func(*nc)
                elif isinstance(m, MatMul):
                    new_obj = m.func(c_mul, *nc, evaluate=False)
                else:
                    new_obj = m.func(c_mul, m.func(*nc), evaluate=False)
                opt_subs[m] = new_obj
            if len(c) > 1:
                commutative_muls.add(c_mul)
    match_common_args(Add, adds, opt_subs)
    match_common_args(Mul, commutative_muls, opt_subs)
    return opt_subs