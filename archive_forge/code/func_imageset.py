from typing import Any, Callable
from functools import reduce
from collections import defaultdict
import inspect
from sympy.core.kind import Kind, UndefinedKind, NumberKind
from sympy.core.basic import Basic
from sympy.core.containers import Tuple, TupleKind
from sympy.core.decorators import sympify_method_args, sympify_return
from sympy.core.evalf import EvalfMixin
from sympy.core.expr import Expr
from sympy.core.function import Lambda
from sympy.core.logic import (FuzzyBool, fuzzy_bool, fuzzy_or, fuzzy_and,
from sympy.core.numbers import Float, Integer
from sympy.core.operations import LatticeOp
from sympy.core.parameters import global_parameters
from sympy.core.relational import Eq, Ne, is_lt
from sympy.core.singleton import Singleton, S
from sympy.core.sorting import ordered
from sympy.core.symbol import symbols, Symbol, Dummy, uniquely_named_symbol
from sympy.core.sympify import _sympify, sympify, _sympy_converter
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import Max, Min
from sympy.logic.boolalg import And, Or, Not, Xor, true, false
from sympy.utilities.decorator import deprecated
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import (iproduct, sift, roundrobin, iterable,
from sympy.utilities.misc import func_name, filldedent
from mpmath import mpi, mpf
from mpmath.libmp.libmpf import prec_to_dps
def imageset(*args):
    """
    Return an image of the set under transformation ``f``.

    Explanation
    ===========

    If this function cannot compute the image, it returns an
    unevaluated ImageSet object.

    .. math::
        \\{ f(x) \\mid x \\in \\mathrm{self} \\}

    Examples
    ========

    >>> from sympy import S, Interval, imageset, sin, Lambda
    >>> from sympy.abc import x

    >>> imageset(x, 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(lambda x: 2*x, Interval(0, 2))
    Interval(0, 4)

    >>> imageset(Lambda(x, sin(x)), Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))

    >>> imageset(sin, Interval(-2, 1))
    ImageSet(Lambda(x, sin(x)), Interval(-2, 1))
    >>> imageset(lambda y: x + y, Interval(-2, 1))
    ImageSet(Lambda(y, x + y), Interval(-2, 1))

    Expressions applied to the set of Integers are simplified
    to show as few negatives as possible and linear expressions
    are converted to a canonical form. If this is not desirable
    then the unevaluated ImageSet should be used.

    >>> imageset(x, -2*x + 5, S.Integers)
    ImageSet(Lambda(x, 2*x + 1), Integers)

    See Also
    ========

    sympy.sets.fancysets.ImageSet

    """
    from .fancysets import ImageSet
    from .setexpr import set_function
    if len(args) < 2:
        raise ValueError('imageset expects at least 2 args, got: %s' % len(args))
    if isinstance(args[0], (Symbol, tuple)) and len(args) > 2:
        f = Lambda(args[0], args[1])
        set_list = args[2:]
    else:
        f = args[0]
        set_list = args[1:]
    if isinstance(f, Lambda):
        pass
    elif callable(f):
        nargs = getattr(f, 'nargs', {})
        if nargs:
            if len(nargs) != 1:
                raise NotImplementedError(filldedent('\n                    This function can take more than 1 arg\n                    but the potentially complicated set input\n                    has not been analyzed at this point to\n                    know its dimensions. TODO\n                    '))
            N = nargs.args[0]
            if N == 1:
                s = 'x'
            else:
                s = [Symbol('x%i' % i) for i in range(1, N + 1)]
        else:
            s = inspect.signature(f).parameters
        dexpr = _sympify(f(*[Dummy() for i in s]))
        var = tuple((uniquely_named_symbol(Symbol(i), dexpr) for i in s))
        f = Lambda(var, f(*var))
    else:
        raise TypeError(filldedent("\n            expecting lambda, Lambda, or FunctionClass,\n            not '%s'." % func_name(f)))
    if any((not isinstance(s, Set) for s in set_list)):
        name = [func_name(s) for s in set_list]
        raise ValueError('arguments after mapping should be sets, not %s' % name)
    if len(set_list) == 1:
        set = set_list[0]
        try:
            r = set_function(f, set)
            if r is None:
                raise TypeError
            if not r:
                return r
        except TypeError:
            r = ImageSet(f, set)
        if isinstance(r, ImageSet):
            f, set = r.args
        if f.variables[0] == f.expr:
            return set
        if isinstance(set, ImageSet):
            if len(set.lamda.variables) == 1 and len(f.variables) == 1:
                x = set.lamda.variables[0]
                y = f.variables[0]
                return imageset(Lambda(x, f.expr.subs(y, set.lamda.expr)), *set.base_sets)
        if r is not None:
            return r
    return ImageSet(f, *set_list)