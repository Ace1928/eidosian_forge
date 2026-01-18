from twisted.protocols import pcp
from twisted.trial import unittest
def testStop(self):
    self.proxy.stopProducing()
    self.assertTrue(self.parentProducer.stopped)