from twisted.protocols import pcp
from twisted.trial import unittest
def testMergeWrites(self):
    self.proxy.write('hello ')
    self.proxy.write('sunshine')
    self.proxy.resumeProducing()
    nwrites = len(self.underlying._writes)
    self.assertEqual(nwrites, 1, 'Pull resulted in %d writes instead of 1.' % (nwrites,))
    self.assertEqual(self.underlying.getvalue(), 'hello sunshine')