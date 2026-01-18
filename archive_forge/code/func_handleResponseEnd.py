from urllib.parse import quote as urlquote, urlparse, urlunparse
from twisted.internet import reactor
from twisted.internet.protocol import ClientFactory
from twisted.web.http import _QUEUED_SENTINEL, HTTPChannel, HTTPClient, Request
from twisted.web.resource import Resource
from twisted.web.server import NOT_DONE_YET
def handleResponseEnd(self):
    """
        Finish the original request, indicating that the response has been
        completely written to it, and disconnect the outgoing transport.
        """
    if not self._finished:
        self._finished = True
        self.father.finish()
        self.transport.loseConnection()