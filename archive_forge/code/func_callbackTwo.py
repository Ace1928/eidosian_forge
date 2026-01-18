from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def callbackTwo(result):
    results2.append(result)
    return 2