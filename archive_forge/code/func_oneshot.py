from ast import parse
import codecs
import collections
import operator
import os
import re
import timeit
from .compat import importlib_metadata_get
def oneshot(*args, **kw):
    result = self.fget(obj, *args, **kw)

    def memo(*a, **kw):
        return result
    memo.__name__ = self.__name__
    memo.__doc__ = self.__doc__
    obj.__dict__[self.__name__] = memo
    return result