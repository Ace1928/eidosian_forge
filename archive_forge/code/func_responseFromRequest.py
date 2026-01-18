import socket
import time
import warnings
from collections import OrderedDict
from typing import Dict, List
from zope.interface import Interface, implementer
from twisted import cred
from twisted.internet import defer, protocol, reactor
from twisted.protocols import basic
from twisted.python import log
def responseFromRequest(self, code, request):
    """
        Create a response to a request message.
        """
    response = Response(code)
    for name in ('via', 'to', 'from', 'call-id', 'cseq'):
        response.headers[name] = request.headers.get(name, [])[:]
    return response