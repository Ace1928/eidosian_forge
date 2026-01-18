import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def runFromAction(self, **kwargs):
    if self._disconnected:
        return None
    return self(**kwargs)