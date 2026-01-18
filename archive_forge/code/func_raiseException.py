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
def raiseException(self) -> NoReturn:
    """
        raise the original exception, preserving traceback
        information if available.
        """
    raise self.value.with_traceback(self.tb)