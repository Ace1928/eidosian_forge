from __future__ import absolute_import
import sys
from types import FunctionType
from future.utils import PY3, PY26
def newsuper(typ=_SENTINEL, type_or_obj=_SENTINEL, framedepth=1):
    """Like builtin super(), but capable of magic.

    This acts just like the builtin super() function, but if called
    without any arguments it attempts to infer them at runtime.
    """
    if typ is _SENTINEL:
        f = sys._getframe(framedepth)
        try:
            type_or_obj = f.f_locals[f.f_code.co_varnames[0]]
        except (IndexError, KeyError):
            raise RuntimeError('super() used in a function with no args')
        try:
            typ = find_owner(type_or_obj, f.f_code)
        except (AttributeError, RuntimeError, TypeError):
            try:
                typ = find_owner(type_or_obj.__class__, f.f_code)
            except AttributeError:
                raise RuntimeError('super() used with an old-style class')
            except TypeError:
                raise RuntimeError('super() called outside a method')
    if type_or_obj is not _SENTINEL:
        return _builtin_super(typ, type_or_obj)
    return _builtin_super(typ)