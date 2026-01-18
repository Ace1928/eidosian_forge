from collections import deque
from typing import Deque, Dict, List, Optional, Tuple
from twisted.internet import defer
from twisted.internet.base import ReactorBase
from twisted.internet.defer import Deferred
from twisted.internet.endpoints import HostnameEndpoint
from twisted.python.failure import Failure
from twisted.web.client import (
from twisted.web.error import SchemeNotSupported
from scrapy.core.downloader.contextfactory import AcceptableProtocolsContextFactory
from scrapy.core.http2.protocol import H2ClientFactory, H2ClientProtocol
from scrapy.http.request import Request
from scrapy.settings import Settings
from scrapy.spiders import Spider
def get_endpoint(self, uri: URI):
    return self.endpoint_factory.endpointForURI(self._proxy_uri)