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
def getTracebackObject(self):
    """
        Get an object that represents this Failure's stack that can be passed
        to traceback.extract_tb.

        If the original traceback object is still present, return that. If this
        traceback object has been lost but we still have the information,
        return a fake traceback object (see L{_Traceback}). If there is no
        traceback information at all, return None.
        """
    if self.tb is not None:
        return self.tb
    elif len(self.frames) > 0:
        return _Traceback(self.stack, self.frames)
    else:
        return None