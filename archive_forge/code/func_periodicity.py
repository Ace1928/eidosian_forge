from .accumulationbounds import AccumBounds, AccumulationBounds # noqa: F401
from .singularities import singularities
from sympy.core import Pow, S
from sympy.core.function import diff, expand_mul
from sympy.core.kind import NumberKind
from sympy.core.mod import Mod
from sympy.core.numbers import equal_valued
from sympy.core.relational import Relational
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import _sympify
from sympy.functions.elementary.complexes import Abs, im, re
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.elementary.trigonometric import (
from sympy.polys.polytools import degree, lcm_list
from sympy.sets.sets import (Interval, Intersection, FiniteSet, Union,
from sympy.sets.fancysets import ImageSet
from sympy.utilities import filldedent
from sympy.utilities.iterables import iterable
def periodicity(f, symbol, check=False):
    """
    Tests the given function for periodicity in the given symbol.

    Parameters
    ==========

    f : :py:class:`~.Expr`
        The concerned function.
    symbol : :py:class:`~.Symbol`
        The variable for which the period is to be determined.
    check : bool, optional
        The flag to verify whether the value being returned is a period or not.

    Returns
    =======

    period
        The period of the function is returned.
        ``None`` is returned when the function is aperiodic or has a complex period.
        The value of $0$ is returned as the period of a constant function.

    Raises
    ======

    NotImplementedError
        The value of the period computed cannot be verified.


    Notes
    =====

    Currently, we do not support functions with a complex period.
    The period of functions having complex periodic values such
    as ``exp``, ``sinh`` is evaluated to ``None``.

    The value returned might not be the "fundamental" period of the given
    function i.e. it may not be the smallest periodic value of the function.

    The verification of the period through the ``check`` flag is not reliable
    due to internal simplification of the given expression. Hence, it is set
    to ``False`` by default.

    Examples
    ========
    >>> from sympy import periodicity, Symbol, sin, cos, tan, exp
    >>> x = Symbol('x')
    >>> f = sin(x) + sin(2*x) + sin(3*x)
    >>> periodicity(f, x)
    2*pi
    >>> periodicity(sin(x)*cos(x), x)
    pi
    >>> periodicity(exp(tan(2*x) - 1), x)
    pi/2
    >>> periodicity(sin(4*x)**cos(2*x), x)
    pi
    >>> periodicity(exp(x), x)
    """
    if symbol.kind is not NumberKind:
        raise NotImplementedError('Cannot use symbol of kind %s' % symbol.kind)
    temp = Dummy('x', real=True)
    f = f.subs(symbol, temp)
    symbol = temp

    def _check(orig_f, period):
        """Return the checked period or raise an error."""
        new_f = orig_f.subs(symbol, symbol + period)
        if new_f.equals(orig_f):
            return period
        else:
            raise NotImplementedError(filldedent('\n                The period of the given function cannot be verified.\n                When `%s` was replaced with `%s + %s` in `%s`, the result\n                was `%s` which was not recognized as being the same as\n                the original function.\n                So either the period was wrong or the two forms were\n                not recognized as being equal.\n                Set check=False to obtain the value.' % (symbol, symbol, period, orig_f, new_f)))
    orig_f = f
    period = None
    if isinstance(f, Relational):
        f = f.lhs - f.rhs
    f = f.simplify()
    if symbol not in f.free_symbols:
        return S.Zero
    if isinstance(f, TrigonometricFunction):
        try:
            period = f.period(symbol)
        except NotImplementedError:
            pass
    if isinstance(f, Abs):
        arg = f.args[0]
        if isinstance(arg, (sec, csc, cos)):
            arg = sin(arg.args[0])
        period = periodicity(arg, symbol)
        if period is not None and isinstance(arg, sin):
            orig_f = Abs(arg)
            try:
                return _check(orig_f, period / 2)
            except NotImplementedError as err:
                if check:
                    raise NotImplementedError(err)
    if isinstance(f, exp) or (f.is_Pow and f.base == S.Exp1):
        f = Pow(S.Exp1, expand_mul(f.exp))
        if im(f) != 0:
            period_real = periodicity(re(f), symbol)
            period_imag = periodicity(im(f), symbol)
            if period_real is not None and period_imag is not None:
                period = lcim([period_real, period_imag])
    if f.is_Pow and f.base != S.Exp1:
        base, expo = f.args
        base_has_sym = base.has(symbol)
        expo_has_sym = expo.has(symbol)
        if base_has_sym and (not expo_has_sym):
            period = periodicity(base, symbol)
        elif expo_has_sym and (not base_has_sym):
            period = periodicity(expo, symbol)
        else:
            period = _periodicity(f.args, symbol)
    elif f.is_Mul:
        coeff, g = f.as_independent(symbol, as_Add=False)
        if isinstance(g, TrigonometricFunction) or not equal_valued(coeff, 1):
            period = periodicity(g, symbol)
        else:
            period = _periodicity(g.args, symbol)
    elif f.is_Add:
        k, g = f.as_independent(symbol)
        if k is not S.Zero:
            return periodicity(g, symbol)
        period = _periodicity(g.args, symbol)
    elif isinstance(f, Mod):
        a, n = f.args
        if a == symbol:
            period = n
        elif isinstance(a, TrigonometricFunction):
            period = periodicity(a, symbol)
        elif a.is_polynomial(symbol) and degree(a, symbol) == 1 and (symbol not in n.free_symbols):
            period = Abs(n / a.diff(symbol))
    elif isinstance(f, Piecewise):
        pass
    elif period is None:
        from sympy.solvers.decompogen import compogen, decompogen
        g_s = decompogen(f, symbol)
        num_of_gs = len(g_s)
        if num_of_gs > 1:
            for index, g in enumerate(reversed(g_s)):
                start_index = num_of_gs - 1 - index
                g = compogen(g_s[start_index:], symbol)
                if g not in (orig_f, f):
                    period = periodicity(g, symbol)
                    if period is not None:
                        break
    if period is not None:
        if check:
            return _check(orig_f, period)
        return period
    return None