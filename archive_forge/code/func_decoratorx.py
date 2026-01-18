import re
import sys
import inspect
import operator
import itertools
from contextlib import _GeneratorContextManager
from inspect import getfullargspec, iscoroutinefunction, isgeneratorfunction
def decoratorx(caller):
    """
    A version of "decorator" implemented via "exec" and not via the
    Signature object. Use this if you are want to preserve the `.__code__`
    object properties (https://github.com/micheles/decorator/issues/129).
    """

    def dec(func):
        return FunctionMaker.create(func, 'return _call_(_func_, %(shortsignature)s)', dict(_call_=caller, _func_=func), __wrapped__=func, __qualname__=func.__qualname__)
    return dec