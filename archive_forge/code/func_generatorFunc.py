from io import StringIO
from twisted.internet import defer
from twisted.python import log
from twisted.python.reflect import qual
from twisted.spread import flavors, jelly, pb
from twisted.test.iosim import connectedServerAndClient
from twisted.trial import unittest
def generatorFunc():
    try:
        yield None
    except pb.RemoteError as exc:
        exception.append(exc)
    else:
        self.fail('RemoteError not raised')