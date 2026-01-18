from collections import Counter, defaultdict, OrderedDict
from itertools import (
from itertools import product as cartes # noqa: F401
from operator import gt
from sympy.utilities.enumerative import (
from sympy.utilities.misc import as_int
from sympy.utilities.decorator import deprecated
def numbered_symbols(prefix='x', cls=None, start=0, exclude=(), *args, **assumptions):
    """
    Generate an infinite stream of Symbols consisting of a prefix and
    increasing subscripts provided that they do not occur in ``exclude``.

    Parameters
    ==========

    prefix : str, optional
        The prefix to use. By default, this function will generate symbols of
        the form "x0", "x1", etc.

    cls : class, optional
        The class to use. By default, it uses ``Symbol``, but you can also use ``Wild``
        or ``Dummy``.

    start : int, optional
        The start number.  By default, it is 0.

    Returns
    =======

    sym : Symbol
        The subscripted symbols.
    """
    exclude = set(exclude or [])
    if cls is None:
        from sympy.core import Symbol
        cls = Symbol
    while True:
        name = '%s%s' % (prefix, start)
        s = cls(name, *args, **assumptions)
        if s not in exclude:
            yield s
        start += 1