from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
def testBucketParentRate(self) -> None:
    self.parent.rate = 1
    self.child1.add(100)
    self.clock.set(10)
    fit = self.child1.add(100)
    self.assertEqual(10, fit)