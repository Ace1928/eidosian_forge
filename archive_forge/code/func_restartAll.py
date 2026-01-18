from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
def restartAll(self):
    """
        Restart all processes. This is useful for third party management
        services to allow a user to restart servers because of an outside change
        in circumstances -- for example, a new version of a library is
        installed.
        """
    for name in self._processes:
        self.stopProcess(name)