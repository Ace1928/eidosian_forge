from twisted.protocols import htb
from twisted.trial import unittest
from .test_pcp import DummyConsumer
def testBucketSize(self) -> None:
    """
        Testing the size of the bucket.
        """
    b = SomeBucket()
    fit = b.add(1000)
    self.assertEqual(100, fit)