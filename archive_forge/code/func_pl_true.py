from sympy.logic.boolalg import And, Not, conjuncts, to_cnf, BooleanFunction
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.external.importtools import import_module
def pl_true(expr, model=None, deep=False):
    """
    Returns whether the given assignment is a model or not.

    If the assignment does not specify the value for every proposition,
    this may return None to indicate 'not obvious'.

    Parameters
    ==========

    model : dict, optional, default: {}
        Mapping of symbols to boolean values to indicate assignment.
    deep: boolean, optional, default: False
        Gives the value of the expression under partial assignments
        correctly. May still return None to indicate 'not obvious'.


    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.inference import pl_true
    >>> pl_true( A & B, {A: True, B: True})
    True
    >>> pl_true(A & B, {A: False})
    False
    >>> pl_true(A & B, {A: True})
    >>> pl_true(A & B, {A: True}, deep=True)
    >>> pl_true(A >> (B >> A))
    >>> pl_true(A >> (B >> A), deep=True)
    True
    >>> pl_true(A & ~A)
    >>> pl_true(A & ~A, deep=True)
    False
    >>> pl_true(A & B & (~A | ~B), {A: True})
    >>> pl_true(A & B & (~A | ~B), {A: True}, deep=True)
    False

    """
    from sympy.core.symbol import Symbol
    boolean = (True, False)

    def _validate(expr):
        if isinstance(expr, Symbol) or expr in boolean:
            return True
        if not isinstance(expr, BooleanFunction):
            return False
        return all((_validate(arg) for arg in expr.args))
    if expr in boolean:
        return expr
    expr = sympify(expr)
    if not _validate(expr):
        raise ValueError('%s is not a valid boolean expression' % expr)
    if not model:
        model = {}
    model = {k: v for k, v in model.items() if v in boolean}
    result = expr.subs(model)
    if result in boolean:
        return bool(result)
    if deep:
        model = {k: True for k in result.atoms()}
        if pl_true(result, model):
            if valid(result):
                return True
        elif not satisfiable(result):
            return False
    return None