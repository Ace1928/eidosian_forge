from twisted.protocols import pcp
from twisted.trial import unittest
def testUnregister(self):
    self.consumer.unregisterProducer()
    self.assertTrue(self.underlying.unregistered)