from collections import defaultdict
from itertools import chain, combinations, product, permutations
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.function import Application, Derivative
from sympy.core.kind import BooleanKind, NumberKind
from sympy.core.numbers import Number
from sympy.core.operations import LatticeOp
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.sympify import _sympy_converter, _sympify, sympify
from sympy.utilities.iterables import sift, ibin
from sympy.utilities.misc import filldedent
def simplify_logic(expr, form=None, deep=True, force=False, dontcare=None):
    """
    This function simplifies a boolean function to its simplified version
    in SOP or POS form. The return type is an :py:class:`~.Or` or
    :py:class:`~.And` object in SymPy.

    Parameters
    ==========

    expr : Boolean

    form : string (``'cnf'`` or ``'dnf'``) or ``None`` (default).
        If ``'cnf'`` or ``'dnf'``, the simplest expression in the corresponding
        normal form is returned; if ``None``, the answer is returned
        according to the form with fewest args (in CNF by default).

    deep : bool (default ``True``)
        Indicates whether to recursively simplify any
        non-boolean functions contained within the input.

    force : bool (default ``False``)
        As the simplifications require exponential time in the number
        of variables, there is by default a limit on expressions with
        8 variables. When the expression has more than 8 variables
        only symbolical simplification (controlled by ``deep``) is
        made. By setting ``force`` to ``True``, this limit is removed. Be
        aware that this can lead to very long simplification times.

    dontcare : Boolean
        Optimize expression under the assumption that inputs where this
        expression is true are don't care. This is useful in e.g. Piecewise
        conditions, where later conditions do not need to consider inputs that
        are converted by previous conditions. For example, if a previous
        condition is ``And(A, B)``, the simplification of expr can be made
        with don't cares for ``And(A, B)``.

    Examples
    ========

    >>> from sympy.logic import simplify_logic
    >>> from sympy.abc import x, y, z
    >>> b = (~x & ~y & ~z) | ( ~x & ~y & z)
    >>> simplify_logic(b)
    ~x & ~y
    >>> simplify_logic(x | y, dontcare=y)
    x

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Don%27t-care_term

    """
    if form not in (None, 'cnf', 'dnf'):
        raise ValueError('form can be cnf or dnf only')
    expr = sympify(expr)
    if form:
        form_ok = False
        if form == 'cnf':
            form_ok = is_cnf(expr)
        elif form == 'dnf':
            form_ok = is_dnf(expr)
        if form_ok and all((is_literal(a) for a in expr.args)):
            return expr
    from sympy.core.relational import Relational
    if deep:
        variables = expr.atoms(Relational)
        from sympy.simplify.simplify import simplify
        s = tuple(map(simplify, variables))
        expr = expr.xreplace(dict(zip(variables, s)))
    if not isinstance(expr, BooleanFunction):
        return expr
    repl = {}
    undo = {}
    from sympy.core.symbol import Dummy
    variables = expr.atoms(Relational)
    if dontcare is not None:
        dontcare = sympify(dontcare)
        variables.update(dontcare.atoms(Relational))
    while variables:
        var = variables.pop()
        if var.is_Relational:
            d = Dummy()
            undo[d] = var
            repl[var] = d
            nvar = var.negated
            if nvar in variables:
                repl[nvar] = Not(d)
                variables.remove(nvar)
    expr = expr.xreplace(repl)
    if dontcare is not None:
        dontcare = dontcare.xreplace(repl)
    variables = _find_predicates(expr)
    if not force and len(variables) > 8:
        return expr.xreplace(undo)
    if dontcare is not None:
        dcvariables = _find_predicates(dontcare)
        variables.update(dcvariables)
        if not force and len(variables) > 8:
            variables = _find_predicates(expr)
            dontcare = None
    c, v = sift(ordered(variables), lambda x: x in (True, False), binary=True)
    variables = c + v
    c = [1 if i == True else 0 for i in c]
    truthtable = _get_truthtable(v, expr, c)
    if dontcare is not None:
        dctruthtable = _get_truthtable(v, dontcare, c)
        truthtable = [t for t in truthtable if t not in dctruthtable]
    else:
        dctruthtable = []
    big = len(truthtable) >= 2 ** (len(variables) - 1)
    if form == 'dnf' or (form is None and big):
        return _sop_form(variables, truthtable, dctruthtable).xreplace(undo)
    return POSform(variables, truthtable, dctruthtable).xreplace(undo)