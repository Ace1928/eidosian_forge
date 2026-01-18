from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def stopit(result):
    callbackPhases.append(result)
    self.cooperator.stop()
    callbackPhases.append('done')