from twisted.protocols import pcp
from twisted.trial import unittest
def testPauseIntercept(self):
    self.proxy.pauseProducing()
    self.assertFalse(self.parentProducer.paused)