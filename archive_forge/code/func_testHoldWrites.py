from twisted.protocols import pcp
from twisted.trial import unittest
def testHoldWrites(self):
    self.proxy.write('hello')
    self.assertFalse(self.underlying.getvalue(), 'Pulling Consumer got data before it pulled.')