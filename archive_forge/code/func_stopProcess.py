from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
def stopProcess(self, name):
    """
        @param name: The name of the process to be stopped
        """
    if name not in self._processes:
        raise KeyError(f'Unrecognized process name: {name}')
    proto = self.protocols.get(name, None)
    if proto is not None:
        proc = proto.transport
        try:
            proc.signalProcess('TERM')
        except error.ProcessExitedAlready:
            pass
        else:
            self.murder[name] = self._reactor.callLater(self.killTime, self._forceStopProcess, proc)