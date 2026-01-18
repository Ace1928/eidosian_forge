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
def request_pty_req(self, data):
    if not self.session:
        self.session = ISession(self.avatar)
    term, windowSize, modes = parseRequest_pty_req(data)
    log.info('Handling pty request: {term!r} {windowSize!r}', term=term, windowSize=windowSize)
    try:
        self.session.getPty(term, windowSize, modes)
    except Exception:
        log.failure('Error handling pty request')
        return 0
    else:
        return 1