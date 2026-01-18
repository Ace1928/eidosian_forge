from twisted.internet import defer, reactor, task
from twisted.trial import unittest
def myiter():
    for i in range(100):
        output.append(i)
        if i == 9:
            _TPF.stopped = True
        yield i