from time import time
from typing import Optional
from zope.interface import Interface, implementer
from twisted.protocols import pcp
def getBucketFor(self, *a, **kw):
    """
        Find or create a L{Bucket} corresponding to the provided parameters.

        Any parameters are passed on to L{getBucketKey}, from them it
        decides which bucket you get.

        @returntype: L{Bucket}
        """
    if self.sweepInterval is not None and time() - self.lastSweep > self.sweepInterval:
        self.sweep()
    if self.parentFilter:
        parentBucket = self.parentFilter.getBucketFor(self, *a, **kw)
    else:
        parentBucket = None
    key = self.getBucketKey(*a, **kw)
    bucket = self.buckets.get(key)
    if bucket is None:
        bucket = self.bucketFactory(parentBucket)
        self.buckets[key] = bucket
    return bucket