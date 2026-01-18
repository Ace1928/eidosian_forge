from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def ignoreUnhandled(failure):
    failure.trap(UnhandledException)
    return None