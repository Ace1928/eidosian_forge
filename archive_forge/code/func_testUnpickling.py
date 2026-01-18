import pickle
import sys
from unittest import skipIf
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase
def testUnpickling(self):
    lockPickle = b'ctwisted.python.threadable\nunpickle_lock\np0\n(tp1\nRp2\n.'
    lock = pickle.loads(lockPickle)
    newPickle = pickle.dumps(lock, 2)
    pickle.loads(newPickle)