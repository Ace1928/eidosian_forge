from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
def testBucketDrain(self) -> None:
    """
        Testing the bucket's drain rate.
        """
    b = SomeBucket()
    fit = b.add(1000)
    self.clock.set(10)
    fit = b.add(1000)
    self.assertEqual(20, fit)