from twisted.protocols import pcp
from twisted.trial import unittest
def testLateWrite(self):
    self.proxy.resumeProducing()
    self.proxy.write('data')
    self.assertEqual(self.underlying.getvalue(), 'data')