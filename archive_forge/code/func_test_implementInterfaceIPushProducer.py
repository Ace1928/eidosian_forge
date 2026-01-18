from zope.interface.verify import verifyClass
from twisted.internet.abstract import FileDescriptor
from twisted.internet.interfaces import IPushProducer
from twisted.trial.unittest import SynchronousTestCase
def test_implementInterfaceIPushProducer(self):
    """
        L{FileDescriptor} should implement L{IPushProducer}.
        """
    self.assertTrue(verifyClass(IPushProducer, FileDescriptor))