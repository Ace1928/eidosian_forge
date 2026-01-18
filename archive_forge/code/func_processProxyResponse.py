import ipaddress
import logging
import re
from contextlib import suppress
from io import BytesIO
from time import time
from urllib.parse import urldefrag, urlunparse
from twisted.internet import defer, protocol, ssl
from twisted.internet.endpoints import TCP4ClientEndpoint
from twisted.internet.error import TimeoutError
from twisted.python.failure import Failure
from twisted.web.client import (
from twisted.web.http import PotentialDataLoss, _DataLoss
from twisted.web.http_headers import Headers as TxHeaders
from twisted.web.iweb import UNKNOWN_LENGTH, IBodyProducer
from zope.interface import implementer
from scrapy import signals
from scrapy.core.downloader.contextfactory import load_context_factory_from_settings
from scrapy.core.downloader.webclient import _parse
from scrapy.exceptions import StopDownload
from scrapy.http import Headers
from scrapy.responsetypes import responsetypes
from scrapy.utils.python import to_bytes, to_unicode
def processProxyResponse(self, rcvd_bytes):
    """Processes the response from the proxy. If the tunnel is successfully
        created, notifies the client that we are ready to send requests. If not
        raises a TunnelError.
        """
    self._connectBuffer += rcvd_bytes
    if b'\r\n\r\n' not in self._connectBuffer:
        return
    self._protocol.dataReceived = self._protocolDataReceived
    respm = TunnelingTCP4ClientEndpoint._responseMatcher.match(self._connectBuffer)
    if respm and int(respm.group('status')) == 200:
        sslOptions = self._contextFactory.creatorForNetloc(self._tunneledHost, self._tunneledPort)
        self._protocol.transport.startTLS(sslOptions, self._protocolFactory)
        self._tunnelReadyDeferred.callback(self._protocol)
    else:
        if respm:
            extra = {'status': int(respm.group('status')), 'reason': respm.group('reason').strip()}
        else:
            extra = rcvd_bytes[:self._truncatedLength]
        self._tunnelReadyDeferred.errback(TunnelError(f'Could not open CONNECT tunnel with proxy {self._host}:{self._port} [{extra!r}]'))