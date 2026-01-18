from collections import defaultdict, namedtuple, OrderedDict
from functools import wraps
from itertools import product
import os
import types
import warnings
from .. import __url__
from .deprecation import Deprecation
def defaultnamedtuple(typename, field_names, defaults=()):
    """Generates a new subclass of tuple with default values.

    Parameters
    ----------
    typename : string
        The name of the class.
    field_names : str or iterable
        An iterable of splitable string.
    defaults : iterable
        Default values for ``field_names``, counting ``[-len(defaults):]``.

    Examples
    --------
    >>> Body = defaultnamedtuple('Body', 'x y z density', (1.0,))
    >>> Body.__doc__
    'Body(x, y, z, density)'
    >>> b = Body(10, z=3, y=5)
    >>> b._asdict() == dict(x=10, y=5, z=3, density=1.0)
    True

    Returns
    -------
    A new tuple subclass named ``typename``

    """
    Tuple = namedtuple(typename, field_names)
    Tuple.__new__.__defaults__ = (None,) * len(Tuple._fields)
    if isinstance(defaults, Mapping):
        Tuple.__new__.__defaults__ = tuple(Tuple(**defaults))
    else:
        nmissing = len(Tuple._fields) - len(defaults)
        defaults = (None,) * nmissing + tuple(defaults)
        Tuple.__new__.__defaults__ = tuple(Tuple(*defaults))
    return Tuple