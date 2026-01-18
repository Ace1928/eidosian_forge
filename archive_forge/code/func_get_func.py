import inspect
import textwrap
import re
import pydoc
from warnings import warn
from collections import namedtuple
from collections.abc import Callable, Mapping
import copy
import sys
def get_func(self):
    func_name = getattr(self._f, '__name__', self.__class__.__name__)
    if inspect.isclass(self._f):
        func = getattr(self._f, '__call__', self._f.__init__)
    else:
        func = self._f
    return (func, func_name)