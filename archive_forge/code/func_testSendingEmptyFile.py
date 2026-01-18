from io import BytesIO
from twisted.internet import abstract, defer, protocol
from twisted.protocols import basic, loopback
from twisted.trial import unittest
def testSendingEmptyFile(self) -> None:
    fileSender = basic.FileSender()
    consumer = abstract.FileDescriptor()
    consumer.connected = 1
    emptyFile = BytesIO(b'')
    d = fileSender.beginFileTransfer(emptyFile, consumer, lambda x: x)
    self.assertIsNone(consumer.producer)
    self.assertTrue(d.called, 'producer unregistered with deferred being called')