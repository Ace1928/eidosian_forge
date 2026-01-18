import pickle
import sys
from unittest import skipIf
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase
@skipIf(threadingSkip, 'Platform does not support threads')
def testPickling(self):
    lock = threadable.XLock()
    lockType = type(lock)
    lockPickle = pickle.dumps(lock)
    newLock = pickle.loads(lockPickle)
    self.assertIsInstance(newLock, lockType)