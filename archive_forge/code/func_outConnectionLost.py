import os
import signal
import struct
import sys
from zope.interface import implementer
from twisted.conch.interfaces import (
from twisted.conch.ssh import channel, common, connection
from twisted.internet import interfaces, protocol
from twisted.logger import Logger
from twisted.python.compat import networkString
def outConnectionLost(self):
    """
        EOF should only be sent when both STDOUT and STDERR have been closed.
        """
    if self.lostOutOrErrFlag:
        self.session.conn.sendEOF(self.session)
    else:
        self.lostOutOrErrFlag = True