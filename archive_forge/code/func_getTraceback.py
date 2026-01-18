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
def getTraceback(self, elideFrameworkCode: int=0, detail: str='default') -> str:
    io = StringIO()
    self.printTraceback(file=io, elideFrameworkCode=elideFrameworkCode, detail=detail)
    return io.getvalue()