import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbClientAndServerStarted(ignored):
    server.transport.write(b'write to port no one is listening to', ('127.0.0.1', 80))
    client.transport.write(test_data_to_send, ('127.0.0.1', serverPort._realPortNumber))