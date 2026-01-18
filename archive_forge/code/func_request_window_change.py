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
def request_window_change(self, data):
    if not self.session:
        self.session = ISession(self.avatar)
    winSize = parseRequest_window_change(data)
    try:
        self.session.windowChanged(winSize)
    except Exception:
        log.failure('Error changing window size')
        return 0
    else:
        return 1