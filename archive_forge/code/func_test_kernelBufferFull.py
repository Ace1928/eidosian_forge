from zope.interface.verify import verifyClass
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IPushProducer
from twisted.trial.unittest import SynchronousTestCase
def test_kernelBufferFull(self):
    """
        When L{FileDescriptor.writeSomeData} returns C{0} to indicate no more
        data can be written immediately, L{FileDescriptor.doWrite} returns
        L{None}.
        """
    descriptor = MemoryFile()
    descriptor.write(b'hello, world')
    self.assertIsNone(descriptor.doWrite())