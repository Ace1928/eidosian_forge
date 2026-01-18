from sympy.core.sympify import sympify
from sympy.core import (S, Pow, Dummy, pi, Expr, Wild, Mul, Equality,
from sympy.core.containers import Tuple
from sympy.core.function import (Lambda, expand_complex, AppliedUndef,
from sympy.core.mod import Mod
from sympy.core.numbers import igcd, I, Number, Rational, oo, ilcm
from sympy.core.power import integer_log
from sympy.core.relational import Eq, Ne, Relational
from sympy.core.sorting import default_sort_key, ordered
from sympy.core.symbol import Symbol, _uniquely_named_symbol
from sympy.core.sympify import _sympify
from sympy.polys.matrices.linsolve import _linear_eq_to_dict
from sympy.polys.polyroots import UnsolvableFactorError
from sympy.simplify.simplify import simplify, fraction, trigsimp, nsimplify
from sympy.simplify import powdenest, logcombine
from sympy.functions import (log, tan, cot, sin, cos, sec, csc, exp,
from sympy.functions.elementary.complexes import Abs, arg, re, im
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.miscellaneous import real_root
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.logic.boolalg import And, BooleanTrue
from sympy.sets import (FiniteSet, imageset, Interval, Intersection,
from sympy.sets.sets import Set, ProductSet
from sympy.matrices import zeros, Matrix, MatrixBase
from sympy.ntheory import totient
from sympy.ntheory.factor_ import divisors
from sympy.ntheory.residue_ntheory import discrete_log, nthroot_mod
from sympy.polys import (roots, Poly, degree, together, PolynomialError,
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polytools import invert, groebner, poly
from sympy.polys.solvers import (sympy_eqs_to_ring, solve_lin_sys,
from sympy.polys.matrices.linsolve import _linsolve
from sympy.solvers.solvers import (checksol, denoms, unrad,
from sympy.solvers.polysys import solve_poly_system
from sympy.utilities import filldedent
from sympy.utilities.iterables import (numbered_symbols, has_dups,
from sympy.calculus.util import periodicity, continuous_domain, function_range
from types import GeneratorType
def substitution(system, symbols, result=[{}], known_symbols=[], exclude=[], all_symbols=None):
    """
    Solves the `system` using substitution method. It is used in
    :func:`~.nonlinsolve`. This will be called from :func:`~.nonlinsolve` when any
    equation(s) is non polynomial equation.

    Parameters
    ==========

    system : list of equations
        The target system of equations
    symbols : list of symbols to be solved.
        The variable(s) for which the system is solved
    known_symbols : list of solved symbols
        Values are known for these variable(s)
    result : An empty list or list of dict
        If No symbol values is known then empty list otherwise
        symbol as keys and corresponding value in dict.
    exclude : Set of expression.
        Mostly denominator expression(s) of the equations of the system.
        Final solution should not satisfy these expressions.
    all_symbols : known_symbols + symbols(unsolved).

    Returns
    =======

    A FiniteSet of ordered tuple of values of `all_symbols` for which the
    `system` has solution. Order of values in the tuple is same as symbols
    present in the parameter `all_symbols`. If parameter `all_symbols` is None
    then same as symbols present in the parameter `symbols`.

    Please note that general FiniteSet is unordered, the solution returned
    here is not simply a FiniteSet of solutions, rather it is a FiniteSet of
    ordered tuple, i.e. the first & only argument to FiniteSet is a tuple of
    solutions, which is ordered, & hence the returned solution is ordered.

    Also note that solution could also have been returned as an ordered tuple,
    FiniteSet is just a wrapper `{}` around the tuple. It has no other
    significance except for the fact it is just used to maintain a consistent
    output format throughout the solveset.

    Raises
    ======

    ValueError
        The input is not valid.
        The symbols are not given.
    AttributeError
        The input symbols are not :class:`~.Symbol` type.

    Examples
    ========

    >>> from sympy import symbols, substitution
    >>> x, y = symbols('x, y', real=True)
    >>> substitution([x + y], [x], [{y: 1}], [y], set([]), [x, y])
    {(-1, 1)}

    * When you want a soln not satisfying $x + 1 = 0$

    >>> substitution([x + y], [x], [{y: 1}], [y], set([x + 1]), [y, x])
    EmptySet
    >>> substitution([x + y], [x], [{y: 1}], [y], set([x - 1]), [y, x])
    {(1, -1)}
    >>> substitution([x + y - 1, y - x**2 + 5], [x, y])
    {(-3, 4), (2, -1)}

    * Returns both real and complex solution

    >>> x, y, z = symbols('x, y, z')
    >>> from sympy import exp, sin
    >>> substitution([exp(x) - sin(y), y**2 - 4], [x, y])
    {(ImageSet(Lambda(_n, I*(2*_n*pi + pi) + log(sin(2))), Integers), -2),
     (ImageSet(Lambda(_n, 2*_n*I*pi + log(sin(2))), Integers), 2)}

    >>> eqs = [z**2 + exp(2*x) - sin(y), -3 + exp(-y)]
    >>> substitution(eqs, [y, z])
    {(-log(3), -sqrt(-exp(2*x) - sin(log(3)))),
     (-log(3), sqrt(-exp(2*x) - sin(log(3)))),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
      ImageSet(Lambda(_n, -sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers)),
     (ImageSet(Lambda(_n, 2*_n*I*pi - log(3)), Integers),
      ImageSet(Lambda(_n, sqrt(-exp(2*x) + sin(2*_n*I*pi - log(3)))), Integers))}

    """
    if not system:
        return S.EmptySet
    if not symbols:
        msg = 'Symbols must be given, for which solution of the system is to be found.'
        raise ValueError(filldedent(msg))
    if not is_sequence(symbols):
        msg = 'symbols should be given as a sequence, e.g. a list.Not type %s: %s'
        raise TypeError(filldedent(msg % (type(symbols), symbols)))
    if not getattr(symbols[0], 'is_Symbol', False):
        msg = 'Iterable of symbols must be given as second argument, not type %s: %s'
        raise ValueError(filldedent(msg % (type(symbols[0]), symbols[0])))
    if all_symbols is None:
        all_symbols = symbols
    old_result = result
    complements = {}
    intersections = {}
    total_conditionset = -1
    total_solveset_call = -1

    def _unsolved_syms(eq, sort=False):
        """Returns the unsolved symbol present
        in the equation `eq`.
        """
        free = eq.free_symbols
        unsolved = free - set(known_symbols) & set(all_symbols)
        if sort:
            unsolved = list(unsolved)
            unsolved.sort(key=default_sort_key)
        return unsolved
    eqs_in_better_order = list(ordered(system, lambda _: len(_unsolved_syms(_))))

    def add_intersection_complement(result, intersection_dict, complement_dict):
        final_result = []
        for res in result:
            res_copy = res
            for key_res, value_res in res.items():
                intersect_set, complement_set = (None, None)
                for key_sym, value_sym in intersection_dict.items():
                    if key_sym == key_res:
                        intersect_set = value_sym
                for key_sym, value_sym in complement_dict.items():
                    if key_sym == key_res:
                        complement_set = value_sym
                if intersect_set or complement_set:
                    new_value = FiniteSet(value_res)
                    if intersect_set and intersect_set != S.Complexes:
                        new_value = Intersection(new_value, intersect_set)
                    if complement_set:
                        new_value = Complement(new_value, complement_set)
                    if new_value is S.EmptySet:
                        res_copy = None
                        break
                    elif new_value.is_FiniteSet and len(new_value) == 1:
                        res_copy[key_res] = set(new_value).pop()
                    else:
                        res_copy[key_res] = new_value
            if res_copy is not None:
                final_result.append(res_copy)
        return final_result

    def _extract_main_soln(sym, sol, soln_imageset):
        """Separate the Complements, Intersections, ImageSet lambda expr and
        its base_set. This function returns the unmasks sol from different classes
        of sets and also returns the appended ImageSet elements in a
        soln_imageset (dict: where key as unmasked element and value as ImageSet).
        """
        if isinstance(sol, ConditionSet):
            sol = sol.base_set
        if isinstance(sol, Complement):
            complements[sym] = sol.args[1]
            sol = sol.args[0]
        if isinstance(sol, Union):
            sol_args = sol.args
            sol = S.EmptySet
            for sol_arg2 in sol_args:
                if isinstance(sol_arg2, FiniteSet):
                    sol += sol_arg2
                else:
                    sol += FiniteSet(sol_arg2)
        if isinstance(sol, Intersection):
            if sol.args[0] not in (S.Reals, S.Complexes):
                intersections[sym] = sol.args[0]
            sol = sol.args[1]
        if isinstance(sol, ImageSet):
            soln_imagest = sol
            expr2 = sol.lamda.expr
            sol = FiniteSet(expr2)
            soln_imageset[expr2] = soln_imagest
        if not isinstance(sol, FiniteSet):
            sol = FiniteSet(sol)
        return (sol, soln_imageset)

    def _check_exclude(rnew, imgset_yes):
        rnew_ = rnew
        if imgset_yes:
            rnew_copy = rnew.copy()
            dummy_n = imgset_yes[0]
            for key_res, value_res in rnew_copy.items():
                rnew_copy[key_res] = value_res.subs(dummy_n, 0)
            rnew_ = rnew_copy
        try:
            satisfy_exclude = any((checksol(d, rnew_) for d in exclude))
        except TypeError:
            satisfy_exclude = None
        return satisfy_exclude

    def _restore_imgset(rnew, original_imageset, newresult):
        restore_sym = set(rnew.keys()) & set(original_imageset.keys())
        for key_sym in restore_sym:
            img = original_imageset[key_sym]
            rnew[key_sym] = img
        if rnew not in newresult:
            newresult.append(rnew)

    def _append_eq(eq, result, res, delete_soln, n=None):
        u = Dummy('u')
        if n:
            eq = eq.subs(n, 0)
        satisfy = eq if eq in (True, False) else checksol(u, u, eq, minimal=True)
        if satisfy is False:
            delete_soln = True
            res = {}
        else:
            result.append(res)
        return (result, res, delete_soln)

    def _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset, original_imageset, newresult, eq=None):
        """If `rnew` (A dict <symbol: soln>) contains valid soln
        append it to `newresult` list.
        `imgset_yes` is (base, dummy_var) if there was imageset in previously
         calculated result(otherwise empty tuple). `original_imageset` is dict
         of imageset expr and imageset from this result.
        `soln_imageset` dict of imageset expr and imageset of new soln.
        """
        satisfy_exclude = _check_exclude(rnew, imgset_yes)
        delete_soln = False
        if not satisfy_exclude:
            local_n = None
            if imgset_yes:
                local_n = imgset_yes[0]
                base = imgset_yes[1]
                if sym and sol:
                    dummy_list = list(sol.atoms(Dummy))
                    local_n_list = [local_n for i in range(0, len(dummy_list))]
                    dummy_zip = zip(dummy_list, local_n_list)
                    lam = Lambda(local_n, sol.subs(dummy_zip))
                    rnew[sym] = ImageSet(lam, base)
                if eq is not None:
                    newresult, rnew, delete_soln = _append_eq(eq, newresult, rnew, delete_soln, local_n)
            elif eq is not None:
                newresult, rnew, delete_soln = _append_eq(eq, newresult, rnew, delete_soln)
            elif sol in soln_imageset.keys():
                rnew[sym] = soln_imageset[sol]
                _restore_imgset(rnew, original_imageset, newresult)
            else:
                newresult.append(rnew)
        elif satisfy_exclude:
            delete_soln = True
            rnew = {}
        _restore_imgset(rnew, original_imageset, newresult)
        return (newresult, delete_soln)

    def _new_order_result(result, eq):
        first_priority = []
        second_priority = []
        for res in result:
            if not any((isinstance(val, ImageSet) for val in res.values())):
                if eq.subs(res) == 0:
                    first_priority.append(res)
                else:
                    second_priority.append(res)
        if first_priority or second_priority:
            return first_priority + second_priority
        return result

    def _solve_using_known_values(result, solver):
        """Solves the system using already known solution
        (result contains the dict <symbol: value>).
        solver is :func:`~.solveset_complex` or :func:`~.solveset_real`.
        """
        soln_imageset = {}
        total_solvest_call = 0
        total_conditionst = 0
        for index, eq in enumerate(eqs_in_better_order):
            newresult = []
            original_imageset = {}
            imgset_yes = False
            result = _new_order_result(result, eq)
            for res in result:
                got_symbol = set()
                for key_res, value_res in res.items():
                    if isinstance(value_res, ImageSet):
                        res[key_res] = value_res.lamda.expr
                        original_imageset[key_res] = value_res
                        dummy_n = value_res.lamda.expr.atoms(Dummy).pop()
                        base, = value_res.base_sets
                        imgset_yes = (dummy_n, base)
                eq2 = eq.subs(res).expand()
                unsolved_syms = _unsolved_syms(eq2, sort=True)
                if not unsolved_syms:
                    if res:
                        newresult, delete_res = _append_new_soln(res, None, None, imgset_yes, soln_imageset, original_imageset, newresult, eq2)
                        if delete_res:
                            result.remove(res)
                    continue
                depen1, depen2 = eq2.rewrite(Add).as_independent(*unsolved_syms)
                if (depen1.has(Abs) or depen2.has(Abs)) and solver == solveset_complex:
                    continue
                soln_imageset = {}
                for sym in unsolved_syms:
                    not_solvable = False
                    try:
                        soln = solver(eq2, sym)
                        total_solvest_call += 1
                        soln_new = S.EmptySet
                        if isinstance(soln, Complement):
                            complements[sym] = soln.args[1]
                            soln = soln.args[0]
                        if isinstance(soln, Intersection):
                            if soln.args[0] != Interval(-oo, oo):
                                intersections[sym] = soln.args[0]
                            soln_new += soln.args[1]
                        soln = soln_new if soln_new else soln
                        if index > 0 and solver == solveset_real:
                            if not isinstance(soln, (ImageSet, ConditionSet)):
                                soln += solveset_complex(eq2, sym)
                    except (NotImplementedError, ValueError):
                        continue
                    if isinstance(soln, ConditionSet):
                        if soln.base_set in (S.Reals, S.Complexes):
                            soln = S.EmptySet
                            not_solvable = True
                            total_conditionst += 1
                        else:
                            soln = soln.base_set
                    if soln is not S.EmptySet:
                        soln, soln_imageset = _extract_main_soln(sym, soln, soln_imageset)
                    for sol in soln:
                        sol, soln_imageset = _extract_main_soln(sym, sol, soln_imageset)
                        sol = set(sol).pop()
                        free = sol.free_symbols
                        if got_symbol and any((ss in free for ss in got_symbol)):
                            continue
                        rnew = res.copy()
                        for k, v in res.items():
                            if isinstance(v, Expr) and isinstance(sol, Expr):
                                rnew[k] = v.subs(sym, sol)
                        if sol in soln_imageset.keys():
                            imgst = soln_imageset[sol]
                            rnew[sym] = imgst.lamda(*[0 for i in range(0, len(imgst.lamda.variables))])
                        else:
                            rnew[sym] = sol
                        newresult, delete_res = _append_new_soln(rnew, sym, sol, imgset_yes, soln_imageset, original_imageset, newresult)
                        if delete_res:
                            result.remove(res)
                    if not not_solvable:
                        got_symbol.add(sym)
            if newresult:
                result = newresult
        return (result, total_solvest_call, total_conditionst)
    new_result_real, solve_call1, cnd_call1 = _solve_using_known_values(old_result, solveset_real)
    new_result_complex, solve_call2, cnd_call2 = _solve_using_known_values(old_result, solveset_complex)
    total_conditionset += cnd_call1 + cnd_call2
    total_solveset_call += solve_call1 + solve_call2
    if total_conditionset == total_solveset_call and total_solveset_call != -1:
        return _return_conditionset(eqs_in_better_order, all_symbols)
    filtered_complex = []
    for i in list(new_result_complex):
        for j in list(new_result_real):
            if i.keys() != j.keys():
                continue
            if all((a.dummy_eq(b) for a, b in zip(i.values(), j.values()) if not (isinstance(a, int) and isinstance(b, int)))):
                break
        else:
            filtered_complex.append(i)
    result = new_result_real + filtered_complex
    result_all_variables = []
    result_infinite = []
    for res in result:
        if not res:
            continue
        if len(res) < len(all_symbols):
            solved_symbols = res.keys()
            unsolved = list(filter(lambda x: x not in solved_symbols, all_symbols))
            for unsolved_sym in unsolved:
                res[unsolved_sym] = unsolved_sym
            result_infinite.append(res)
        if res not in result_all_variables:
            result_all_variables.append(res)
    if result_infinite:
        result_all_variables = result_infinite
    if intersections or complements:
        result_all_variables = add_intersection_complement(result_all_variables, intersections, complements)
    result = S.EmptySet
    for r in result_all_variables:
        temp = [r[symb] for symb in all_symbols]
        result += FiniteSet(tuple(temp))
    return result