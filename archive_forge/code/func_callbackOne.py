from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def callbackOne(result):
    results1.append(result)
    return 1