from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
def test_bucketEmpty(self) -> None:
    """
        L{htb.Bucket.drip} returns C{True} if the bucket is empty after that drip.
        """
    b = SomeBucket()
    b.add(20)
    self.clock.set(9)
    empty = b.drip()
    self.assertFalse(empty)
    self.clock.set(10)
    empty = b.drip()
    self.assertTrue(empty)