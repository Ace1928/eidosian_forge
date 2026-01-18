import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
def cleanFailure(self):
    """
        Remove references to other objects, replacing them with strings.

        On Python 3, this will also set the C{__traceback__} attribute of the
        exception instance to L{None}.
        """
    self.__dict__ = self.__getstate__()
    if getattr(self.value, '__traceback__', None):
        self.value.__traceback__ = None