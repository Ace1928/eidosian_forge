from twisted.protocols import pcp
from twisted.trial import unittest
def testResumePull(self):
    self.parentProducer.resumed = False
    self.proxy.resumeProducing()
    self.assertTrue(self.parentProducer.resumed)