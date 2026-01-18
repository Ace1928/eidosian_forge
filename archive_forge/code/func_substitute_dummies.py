from collections import defaultdict
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.matrices.dense import zeros
from sympy.printing.str import StrPrinter
from sympy.utilities.iterables import has_dups
def substitute_dummies(expr, new_indices=False, pretty_indices={}):
    """
    Collect terms by substitution of dummy variables.

    Explanation
    ===========

    This routine allows simplification of Add expressions containing terms
    which differ only due to dummy variables.

    The idea is to substitute all dummy variables consistently depending on
    the structure of the term.  For each term, we obtain a sequence of all
    dummy variables, where the order is determined by the index range, what
    factors the index belongs to and its position in each factor.  See
    _get_ordered_dummies() for more information about the sorting of dummies.
    The index sequence is then substituted consistently in each term.

    Examples
    ========

    >>> from sympy import symbols, Function, Dummy
    >>> from sympy.physics.secondquant import substitute_dummies
    >>> a,b,c,d = symbols('a b c d', above_fermi=True, cls=Dummy)
    >>> i,j = symbols('i j', below_fermi=True, cls=Dummy)
    >>> f = Function('f')

    >>> expr = f(a,b) + f(c,d); expr
    f(_a, _b) + f(_c, _d)

    Since a, b, c and d are equivalent summation indices, the expression can be
    simplified to a single term (for which the dummy indices are still summed over)

    >>> substitute_dummies(expr)
    2*f(_a, _b)


    Controlling output:

    By default the dummy symbols that are already present in the expression
    will be reused in a different permutation.  However, if new_indices=True,
    new dummies will be generated and inserted.  The keyword 'pretty_indices'
    can be used to control this generation of new symbols.

    By default the new dummies will be generated on the form i_1, i_2, a_1,
    etc.  If you supply a dictionary with key:value pairs in the form:

        { index_group: string_of_letters }

    The letters will be used as labels for the new dummy symbols.  The
    index_groups must be one of 'above', 'below' or 'general'.

    >>> expr = f(a,b,i,j)
    >>> my_dummies = { 'above':'st', 'below':'uv' }
    >>> substitute_dummies(expr, new_indices=True, pretty_indices=my_dummies)
    f(_s, _t, _u, _v)

    If we run out of letters, or if there is no keyword for some index_group
    the default dummy generator will be used as a fallback:

    >>> p,q = symbols('p q', cls=Dummy)  # general indices
    >>> expr = f(p,q)
    >>> substitute_dummies(expr, new_indices=True, pretty_indices=my_dummies)
    f(_p_0, _p_1)

    """
    if new_indices:
        letters_above = pretty_indices.get('above', '')
        letters_below = pretty_indices.get('below', '')
        letters_general = pretty_indices.get('general', '')
        len_above = len(letters_above)
        len_below = len(letters_below)
        len_general = len(letters_general)

        def _i(number):
            try:
                return letters_below[number]
            except IndexError:
                return 'i_' + str(number - len_below)

        def _a(number):
            try:
                return letters_above[number]
            except IndexError:
                return 'a_' + str(number - len_above)

        def _p(number):
            try:
                return letters_general[number]
            except IndexError:
                return 'p_' + str(number - len_general)
    aboves = []
    belows = []
    generals = []
    dummies = expr.atoms(Dummy)
    if not new_indices:
        dummies = sorted(dummies, key=default_sort_key)
    a = i = p = 0
    for d in dummies:
        assum = d.assumptions0
        if assum.get('above_fermi'):
            if new_indices:
                sym = _a(a)
                a += 1
            l1 = aboves
        elif assum.get('below_fermi'):
            if new_indices:
                sym = _i(i)
                i += 1
            l1 = belows
        else:
            if new_indices:
                sym = _p(p)
                p += 1
            l1 = generals
        if new_indices:
            l1.append(Dummy(sym, **assum))
        else:
            l1.append(d)
    expr = expr.expand()
    terms = Add.make_args(expr)
    new_terms = []
    for term in terms:
        i = iter(belows)
        a = iter(aboves)
        p = iter(generals)
        ordered = _get_ordered_dummies(term)
        subsdict = {}
        for d in ordered:
            if d.assumptions0.get('below_fermi'):
                subsdict[d] = next(i)
            elif d.assumptions0.get('above_fermi'):
                subsdict[d] = next(a)
            else:
                subsdict[d] = next(p)
        subslist = []
        final_subs = []
        for k, v in subsdict.items():
            if k == v:
                continue
            if v in subsdict:
                if subsdict[v] in subsdict:
                    x = Dummy('x')
                    subslist.append((k, x))
                    final_subs.append((x, v))
                else:
                    final_subs.insert(0, (k, v))
            else:
                subslist.append((k, v))
        subslist.extend(final_subs)
        new_terms.append(term.subs(subslist))
    return Add(*new_terms)