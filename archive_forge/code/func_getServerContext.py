from typing import Optional, Sequence, Type
from zope.interface import Interface, implementer
from twisted.internet.defer import Deferred, DeferredList
from twisted.internet.endpoints import (
from twisted.internet.error import ConnectionClosed
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.connectionmixins import (
from twisted.internet.test.reactormixins import ReactorBuilder
from twisted.internet.test.test_tcp import (
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
def getServerContext(self):
    """
        Return a new SSL context suitable for use in a test server.
        """
    pem = self._pem.getContent()
    cert = PrivateCertificate.load(pem, KeyPair.load(pem, FILETYPE_PEM), FILETYPE_PEM)
    return cert.options()