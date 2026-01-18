import itertools
from zope.interface import directlyProvides, providedBy
from twisted.internet import defer, error, reactor, task
from twisted.internet.address import IPv4Address
from twisted.internet.testing import MemoryReactorClock, StringTransport
from twisted.python import failure
from twisted.python.compat import iterbytes
from twisted.test.test_internet import DummyProducer
from twisted.trial import unittest
from twisted.web import http
from twisted.web.test.test_http import (
def buildPriorityFrame(self, streamID, weight, dependsOn=0, exclusive=False):
    """
        Builds a single priority frame.
        """
    f = hyperframe.frame.PriorityFrame(streamID)
    f.depends_on = dependsOn
    f.stream_weight = weight
    f.exclusive = exclusive
    return f