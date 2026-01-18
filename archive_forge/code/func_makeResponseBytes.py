from twisted.internet.testing import MemoryReactor, StringTransportWithDisconnection
from twisted.trial.unittest import TestCase
from twisted.web.proxy import (
from twisted.web.resource import Resource
from twisted.web.server import Site
from twisted.web.test.test_web import DummyRequest
def makeResponseBytes(self, code, message, headers, body):
    lines = [b'HTTP/1.0 ' + str(code).encode('ascii') + b' ' + message]
    for header, values in headers:
        for value in values:
            lines.append(header + b': ' + value)
    lines.extend([b'', body])
    return b'\r\n'.join(lines)