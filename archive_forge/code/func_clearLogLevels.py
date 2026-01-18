from functools import partial
from typing import Dict, Iterable
from zope.interface import Interface, implementer
from constantly import NamedConstant, Names
from ._interfaces import ILogObserver, LogEvent
from ._levels import InvalidLogLevelError, LogLevel
from ._observer import bitbucketLogObserver
def clearLogLevels(self) -> None:
    """
        Clears all log levels to the default.
        """
    self._logLevelsByNamespace.clear()
    self._logLevelsByNamespace[''] = self.defaultLogLevel