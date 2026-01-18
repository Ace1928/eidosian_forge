import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def runFromChangedOrChanging(self, param, value):
    if self._disconnected:
        return None
    oldPropagate = self.parametersNeedRunKwargs
    self.parametersNeedRunKwargs = False
    try:
        ret = self(**{param.name(): value})
    finally:
        self.parametersNeedRunKwargs = oldPropagate
    return ret