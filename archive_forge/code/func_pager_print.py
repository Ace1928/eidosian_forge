import itertools
from sympy.core import S
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import Function
from sympy.core.mul import Mul
from sympy.core.numbers import Number, Rational
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError
from sympy.printing.conventions import requires_partial
from sympy.printing.precedence import PRECEDENCE, precedence, precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.str import sstr
from sympy.utilities.iterables import has_variety
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.printing.pretty.stringpict import prettyForm, stringPict
from sympy.printing.pretty.pretty_symbology import hobj, vobj, xobj, \
def pager_print(expr, **settings):
    """Prints expr using the pager, in pretty form.

    This invokes a pager command using pydoc. Lines are not wrapped
    automatically. This routine is meant to be used with a pager that allows
    sideways scrolling, like ``less -S``.

    Parameters are the same as for ``pretty_print``. If you wish to wrap lines,
    pass ``num_columns=None`` to auto-detect the width of the terminal.

    """
    from pydoc import pager
    from locale import getpreferredencoding
    if 'num_columns' not in settings:
        settings['num_columns'] = 500000
    pager(pretty(expr, **settings).encode(getpreferredencoding()))