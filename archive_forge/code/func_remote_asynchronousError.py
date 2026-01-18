from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def remote_asynchronousError(self):
    """
        Fail asynchronously with a pb.Error exception.
        """
    return defer.fail(AsynchronousError('remote asynchronous error'))