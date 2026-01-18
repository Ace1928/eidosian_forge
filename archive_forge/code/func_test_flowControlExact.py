import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
def test_flowControlExact(self):
    """
        Exactly filling the flow control window still blocks producers.
        """
    f = FrameFactory()
    b = StringTransport()
    a = H2Connection()
    a.requestFactory = DummyProducerHandlerProxy
    requestBytes = f.clientConnectionPreface()
    requestBytes += f.buildSettingsFrame({h2.settings.SettingCodes.INITIAL_WINDOW_SIZE: 5}).serialize()
    requestBytes += buildRequestBytes(self.getRequestHeaders, [], f)
    a.makeConnection(b)
    for byte in iterbytes(requestBytes):
        a.dataReceived(byte)
    stream = a.streams[1]
    request = stream._request.original
    self.assertTrue(stream._producerProducing)
    request.write(b'helloworld')
    self.assertFalse(stream._producerProducing)
    self.assertEqual(request.producer.events, ['pause'])
    request.write(b'h')

    def window_open():
        a.dataReceived(f.buildWindowUpdateFrame(streamID=1, increment=50).serialize())
        self.assertTrue(stream._producerProducing)
        self.assertEqual(request.producer.events, ['pause', 'resume'])
        request.unregisterProducer()
        request.finish()
    windowDefer = task.deferLater(reactor, 0, window_open)

    def validate(streamID):
        frames = framesFromBytes(b.value())
        self.assertTrue('END_STREAM' in frames[-1].flags)
        dataChunks = [f.data for f in frames if isinstance(f, hyperframe.frame.DataFrame)]
        self.assertEqual(dataChunks, [b'hello', b'world', b'h', b''])
    validateDefer = a._streamCleanupCallbacks[1].addCallback(validate)
    return defer.DeferredList([windowDefer, validateDefer])