from __future__ import annotations
from typing import Any
import builtins
import inspect
import keyword
import textwrap
import linecache
from sympy.external import import_module # noqa:F401
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.iterables import (is_sequence, iterable,
from sympy.utilities.misc import filldedent
def implemented_function(symfunc, implementation):
    """ Add numerical ``implementation`` to function ``symfunc``.

    ``symfunc`` can be an ``UndefinedFunction`` instance, or a name string.
    In the latter case we create an ``UndefinedFunction`` instance with that
    name.

    Be aware that this is a quick workaround, not a general method to create
    special symbolic functions. If you want to create a symbolic function to be
    used by all the machinery of SymPy you should subclass the ``Function``
    class.

    Parameters
    ----------
    symfunc : ``str`` or ``UndefinedFunction`` instance
       If ``str``, then create new ``UndefinedFunction`` with this as
       name.  If ``symfunc`` is an Undefined function, create a new function
       with the same name and the implemented function attached.
    implementation : callable
       numerical implementation to be called by ``evalf()`` or ``lambdify``

    Returns
    -------
    afunc : sympy.FunctionClass instance
       function with attached implementation

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.utilities.lambdify import implemented_function
    >>> from sympy import lambdify
    >>> f = implemented_function('f', lambda x: x+1)
    >>> lam_f = lambdify(x, f(x))
    >>> lam_f(4)
    5
    """
    from sympy.core.function import UndefinedFunction
    kwargs = {}
    if isinstance(symfunc, UndefinedFunction):
        kwargs = symfunc._kwargs
        symfunc = symfunc.__name__
    if isinstance(symfunc, str):
        symfunc = UndefinedFunction(symfunc, _imp_=staticmethod(implementation), **kwargs)
    elif not isinstance(symfunc, UndefinedFunction):
        raise ValueError(filldedent('\n            symfunc should be either a string or\n            an UndefinedFunction instance.'))
    return symfunc