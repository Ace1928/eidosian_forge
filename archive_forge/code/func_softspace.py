import sys
from typing import AnyStr, Iterable, Optional
from constantly import NamedConstant
from incremental import Version
from twisted.python.deprecate import deprecatedProperty
from ._levels import LogLevel
from ._logger import Logger
@softspace.setter
def softspace(self, value):
    self._softspace = value