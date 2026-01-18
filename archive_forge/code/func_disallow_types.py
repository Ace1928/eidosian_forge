from __future__ import absolute_import, division, print_function
import functools
from numbers import Integral
from future import utils
def disallow_types(argnums, disallowed_types):
    """
    A decorator that raises a TypeError if any of the given numbered
    arguments is of the corresponding given type (e.g. bytes or unicode
    string).

    For example:

        @disallow_types([0, 1], [unicode, bytes])
        def f(a, b):
            pass

    raises a TypeError when f is called if a unicode object is passed as
    `a` or a bytes object is passed as `b`.

    This also skips over keyword arguments, so

        @disallow_types([0, 1], [unicode, bytes])
        def g(a, b=None):
            pass

    doesn't raise an exception if g is called with only one argument a,
    e.g.:

        g(b'Byte string')

    Example use:

    >>> class newbytes(object):
    ...     @disallow_types([1], [unicode])
    ...     def __add__(self, other):
    ...          pass

    >>> newbytes('1234') + u'1234'      #doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
      ...
    TypeError: can't concat 'bytes' to (unicode) str
    """

    def decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            from .newbytes import newbytes
            from .newint import newint
            from .newstr import newstr
            errmsg = "argument can't be {0}"
            for argnum, mytype in zip(argnums, disallowed_types):
                if isinstance(mytype, str) or isinstance(mytype, bytes):
                    mytype = locals()[mytype]
                if len(args) <= argnum:
                    break
                if type(args[argnum]) == mytype:
                    raise TypeError(errmsg.format(mytype))
            return function(*args, **kwargs)
        return wrapper
    return decorator