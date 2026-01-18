import re
from zope.interface import implementer
from twisted.internet.defer import (
from twisted.internet.error import ConnectionDone
from twisted.internet.interfaces import IConsumer, IPushProducer
from twisted.internet.protocol import Protocol
from twisted.logger import Logger
from twisted.protocols.basic import LineReceiver
from twisted.python.compat import networkString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.reflect import fullyQualifiedName
from twisted.web.http import (
from twisted.web.http_headers import Headers
from twisted.web.iweb import UNKNOWN_LENGTH, IClientRequest, IResponse
def statusReceived(self, status):
    """
        Parse the status line into its components and create a response object
        to keep track of this response's state.
        """
    parts = status.split(b' ', 2)
    if len(parts) == 2:
        version, codeBytes = parts
        phrase = b''
    elif len(parts) == 3:
        version, codeBytes, phrase = parts
    else:
        raise ParseError('wrong number of parts', status)
    try:
        statusCode = int(codeBytes)
    except ValueError:
        raise ParseError('non-integer status code', status)
    self.response = Response._construct(self.parseVersion(version), statusCode, phrase, self.headers, self.transport, self.request)