from twisted.protocols import pcp
from twisted.trial import unittest
def testFinish(self):
    self.consumer.finish()
    self.assertTrue(self.underlying.finished)