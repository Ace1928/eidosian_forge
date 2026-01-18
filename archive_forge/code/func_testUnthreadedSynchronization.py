import pickle
import sys
from unittest import skipIf
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase
def testUnthreadedSynchronization(self):
    o = TestObject()
    for i in range(1000):
        o.aMethod()