import copy
import os
import sys
from io import BytesIO
from xml.dom.minidom import getDOMImplementation
from twisted.internet import address, reactor
from twisted.logger import Logger
from twisted.persisted import styles
from twisted.spread import pb
from twisted.spread.banana import SIZE_LIMIT
from twisted.web import http, resource, server, static, util
from twisted.web.http_headers import Headers
def notConnected(self, msg):
    """I can't connect to a publisher; I'll now reply to all pending
        requests.
        """
    self._log.info('could not connect to distributed web service: {msg}', msg=msg)
    self.waiting = 0
    self.publisher = None
    for request in self.pending:
        request.write('Unable to connect to distributed server.')
        request.finish()
    self.pending = []