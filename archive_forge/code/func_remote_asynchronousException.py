from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def remote_asynchronousException(self):
    """
        Fail asynchronously with a non-pb.Error exception.
        """
    return defer.fail(AsynchronousException('remote asynchronous exception'))