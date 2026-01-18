from collections import OrderedDict, defaultdict
from functools import reduce
from itertools import chain, product
from operator import mul, add
import copy
import math
import warnings
from .util.arithmeticdict import ArithmeticDict
from .util._expr import Expr
from .util.periodic import mass_from_composition
from .util.parsing import (
from .units import default_units, is_quantity, unit_of, to_unitless
from ._util import intdiv
from .util.pyutil import deprecated, DeferredImport, ChemPyDeprecationWarning
def balance_stoichiometry(reactants, products, substances=None, substance_factory=Substance.from_formula, parametric_symbols=None, underdetermined=True, allow_duplicates=False):
    """Balances stoichiometric coefficients of a reaction

    Parameters
    ----------
    reactants : iterable of reactant keys
    products : iterable of product keys
    substances : OrderedDict or string or None
        Mapping reactant/product keys to instances of :class:`Substance`.
    substance_factory : callback
    parametric_symbols : generator of symbols
        Used to generate symbols for parametric solution for
        under-determined system of equations. Default is numbered "x-symbols" starting
        from 1.
    underdetermined : bool
        Allows to find a non-unique solution (in addition to a constant factor
        across all terms). Set to ``False`` to disallow (raise ValueError) on
        e.g. "C + O2 -> CO + CO2". Set to ``None`` if you want the symbols replaced
        so that the coefficients are the smallest possible positive (non-zero) integers.
    allow_duplicates : bool
        If False: raises an exception if keys appear in both ``reactants`` and ``products``.

    Examples
    --------
    >>> ref = {'C2H2': 2, 'O2': 3}, {'CO': 4, 'H2O': 2}
    >>> balance_stoichiometry({'C2H2', 'O2'}, {'CO', 'H2O'}) == ref
    True
    >>> ref2 = {'H2': 1, 'O2': 1}, {'H2O2': 1}
    >>> balance_stoichiometry('H2 O2'.split(), ['H2O2'], 'H2 O2 H2O2') == ref2
    True
    >>> reac, prod = 'CuSCN KIO3 HCl'.split(), 'CuSO4 KCl HCN ICl H2O'.split()
    >>> Reaction(*balance_stoichiometry(reac, prod)).string()
    '4 CuSCN + 7 KIO3 + 14 HCl -> 4 CuSO4 + 7 KCl + 4 HCN + 7 ICl + 5 H2O'
    >>> balance_stoichiometry({'Fe', 'O2'}, {'FeO', 'Fe2O3'}, underdetermined=False)
    Traceback (most recent call last):
        ...
    ValueError: The system was under-determined
    >>> r, p = balance_stoichiometry({'Fe', 'O2'}, {'FeO', 'Fe2O3'})
    >>> list(set.union(*[v.free_symbols for v in r.values()]))
    [x1]
    >>> b = balance_stoichiometry({'Fe', 'O2'}, {'FeO', 'Fe2O3'}, underdetermined=None)
    >>> b == ({'Fe': 3, 'O2': 2}, {'FeO': 1, 'Fe2O3': 1})
    True
    >>> d = balance_stoichiometry({'C', 'CO'}, {'C', 'CO', 'CO2'}, underdetermined=None, allow_duplicates=True)
    >>> d == ({'CO': 2}, {'C': 1, 'CO2': 1})
    True

    Returns
    -------
    balanced reactants : dict
    balanced products : dict

    """
    import sympy
    from sympy import MutableDenseMatrix, gcd, zeros, linsolve, numbered_symbols, nsimplify, Wild, Symbol, Integer, Tuple, preorder_traversal as pre
    _intersect = sorted(set.intersection(*map(set, (reactants, products))))
    if _intersect:
        if allow_duplicates:
            if underdetermined is not None:
                raise NotImplementedError('allow_duplicates currently requires underdetermined=None')
            if set(reactants) == set(products):
                raise ValueError('cannot balance: reactants and products identical')
            for dupl in _intersect:
                try:
                    result = balance_stoichiometry([sp for sp in reactants if sp != dupl], [sp for sp in products if sp != dupl], substances=substances, substance_factory=substance_factory, underdetermined=underdetermined, allow_duplicates=True)
                except Exception:
                    continue
                else:
                    return result
            for perm in product(*[(False, True)] * len(_intersect)):
                r = set(reactants)
                p = set(products)
                for remove_reac, dupl in zip(perm, _intersect):
                    if remove_reac:
                        r.remove(dupl)
                    else:
                        p.remove(dupl)
                try:
                    result = balance_stoichiometry(r, p, substances=substances, substance_factory=substance_factory, parametric_symbols=parametric_symbols, underdetermined=underdetermined, allow_duplicates=False)
                except ValueError:
                    continue
                else:
                    return result
            else:
                raise ValueError('Failed to remove duplicate keys: %s' % _intersect)
        else:
            raise ValueError('Substances on both sides: %s' % str(_intersect))
    if substances is None:
        substances = OrderedDict([(k, substance_factory(k)) for k in chain(reactants, products)])
    if isinstance(substances, str):
        substances = OrderedDict([(k, substance_factory(k)) for k in substances.split()])
    if type(reactants) == set:
        reactants = sorted(reactants)
    if type(products) == set:
        products = sorted(products)
    subst_keys = list(reactants) + list(products)
    cks = Substance.composition_keys(substances.values())
    if parametric_symbols is None:
        parametric_symbols = numbered_symbols('x', start=1, integer=True, positive=True)

    def _get(ck, sk):
        return substances[sk].composition.get(ck, 0) * (-1 if sk in reactants else 1)
    for ck in cks:
        for rk in reactants:
            if substances[rk].composition.get(ck, 0) != 0:
                break
        else:
            any_pos = any((substances[pk].composition.get(ck, 0) > 0 for pk in products))
            any_neg = any((substances[pk].composition.get(ck, 0) < 0 for pk in products))
            if any_pos and any_neg:
                pass
            else:
                raise ValueError("Component '%s' not among reactants" % ck)
        for pk in products:
            if substances[pk].composition.get(ck, 0) != 0:
                break
        else:
            any_pos = any((substances[pk].composition.get(ck, 0) > 0 for pk in reactants))
            any_neg = any((substances[pk].composition.get(ck, 0) < 0 for pk in reactants))
            if any_pos and any_neg:
                pass
            else:
                raise ValueError("Component '%s' not among products" % ck)
    A = MutableDenseMatrix([[_get(ck, sk) for sk in subst_keys] for ck in cks])
    A = nsimplify(A)
    symbs = list(reversed([next(parametric_symbols) for _ in range(len(subst_keys))]))
    sol, = linsolve((A, zeros(len(cks), 1)), symbs)
    try:
        sol = nsimplify(sol)
    except AttributeError:
        pass
    wi = Wild('wi', properties=[lambda k: not k.has(Symbol)])
    cd = reduce(gcd, [1] + [1 / m[wi] for m in map(lambda n: n.match(symbs[-1] / wi), pre(sol)) if m is not None])
    sol = sol.func(*[arg / cd for arg in sol.args])

    def remove(cont, symb, remaining):
        subsd = dict(zip(remaining / symb, remaining))
        cont = cont.func(*[(arg / symb).expand().subs(subsd) for arg in cont.args])
        if cont.has(symb):
            raise ValueError('Bug, please report an issue at https://github.com/bjodah/chempy')
        return cont
    done = False
    for idx, symb in enumerate(symbs):
        for expr in sol:
            iterable = expr.args if expr.is_Add else [expr]
            for term in iterable:
                if term.is_number:
                    done = True
                    break
            if done:
                break
        if done:
            break
        for expr in sol:
            if (expr / symb).is_number:
                sol = remove(sol, symb, MutableDenseMatrix(symbs[idx + 1:]))
                break
    for symb in symbs:
        cd = 1
        for expr in sol:
            iterable = expr.args if expr.is_Add else [expr]
            for term in iterable:
                if term.is_Mul and term.args[0].is_number and (term.args[1] == symb):
                    cd = gcd(cd, term.args[0])
        if cd != 1:
            sol = sol.func(*[arg.subs(symb, symb / cd) for arg in sol.args])
    integer_one = 1
    if underdetermined is integer_one:
        from ._release import __version__
        if int(__version__.split('.')[1]) > 6:
            warnings.warn("Pass underdetermined == None instead of ``1`` (deprecated since 0.7.0, will_be_missing_in='0.9.0')", ChemPyDeprecationWarning)
        underdetermined = None
    if underdetermined is None:
        sol = Tuple(*[Integer(x) for x in _solve_balancing_ilp_pulp(A)])
    fact = gcd(sol)
    sol = MutableDenseMatrix([e / fact for e in sol]).reshape(len(sol), 1)
    sol /= reduce(gcd, sol)
    sol = nsimplify(sol)
    if 0 in sol:
        raise ValueError('Superfluous species given.')
    if underdetermined:
        if any((x == sympy.nan for x in sol)):
            raise ValueError('Failed to balance reaction')
    else:
        for x in sol:
            if len(x.free_symbols) != 0:
                raise ValueError('The system was under-determined')
        if not all((residual == 0 for residual in A * sol)):
            raise ValueError('Failed to balance reaction')

    def _x(k):
        coeff = sol[subst_keys.index(k)]
        return int(coeff) if underdetermined is None else coeff
    return (OrderedDict([(k, _x(k)) for k in reactants]), OrderedDict([(k, _x(k)) for k in products]))