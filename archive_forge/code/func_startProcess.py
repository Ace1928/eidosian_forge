from typing import Dict, List, Optional
import attr
import incremental
from twisted.application import service
from twisted.internet import error, protocol, reactor as _reactor
from twisted.logger import Logger
from twisted.protocols import basic
from twisted.python import deprecate
def startProcess(self, name):
    """
        @param name: The name of the process to be started
        """
    if name in self.protocols:
        return
    process = self._processes[name]
    proto = LoggingProtocol()
    proto.service = self
    proto.name = name
    self.protocols[name] = proto
    self.timeStarted[name] = self._reactor.seconds()
    self._reactor.spawnProcess(proto, process.args[0], process.args, uid=process.uid, gid=process.gid, env=process.env, path=process.cwd)