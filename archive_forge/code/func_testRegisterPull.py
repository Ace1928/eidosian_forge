from twisted.protocols import pcp
from twisted.trial import unittest
def testRegisterPull(self):
    self.proxy.registerProducer(self.parentProducer, False)
    self.assertTrue(self.parentProducer.resumed)