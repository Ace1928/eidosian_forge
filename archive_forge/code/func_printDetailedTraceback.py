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
def printDetailedTraceback(self, file=None, elideFrameworkCode=0):
    """
        Print a traceback with detailed locals and globals information.
        """
    self.printTraceback(file, elideFrameworkCode, detail='verbose')