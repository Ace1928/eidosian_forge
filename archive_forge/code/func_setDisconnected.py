import contextlib
import functools
import inspect
import pydoc
from .. import functions as fn
from . import Parameter
from .parameterTypes import ActionGroupParameter
def setDisconnected(self, disconnected):
    """
        Sets the disconnected state of the runnable, see :meth:`disconnect` and
        :meth:`reconnect` for more information
        """
    oldDisconnect = self._disconnected
    self._disconnected = disconnected
    return oldDisconnect