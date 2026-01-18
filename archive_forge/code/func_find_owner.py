from __future__ import absolute_import
import sys
from types import FunctionType
from future.utils import PY3, PY26
def find_owner(cls, code):
    """Find the class that owns the currently-executing method.
    """
    for typ in cls.__mro__:
        for meth in typ.__dict__.values():
            try:
                while not isinstance(meth, FunctionType):
                    if isinstance(meth, property):
                        meth = meth.fget
                    else:
                        try:
                            meth = meth.__func__
                        except AttributeError:
                            meth = meth.__get__(cls, typ)
            except (AttributeError, TypeError):
                continue
            if meth.func_code is code:
                return typ
    raise TypeError