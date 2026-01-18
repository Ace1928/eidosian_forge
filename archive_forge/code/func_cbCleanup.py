import os
from unittest import skipIf
from twisted.internet import defer, error, interfaces, protocol, reactor, udp
from twisted.internet.defer import Deferred, gatherResults, maybeDeferred
from twisted.python import runtime
from twisted.trial.unittest import TestCase
def cbCleanup(result):
    """
            Disconnect the port we started and pass on whatever was given to us
            in case it was a Failure.
            """
    return defer.maybeDeferred(port.stopListening).addBoth(lambda ign: result)