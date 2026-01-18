import pickle
import sys
from unittest import skipIf
from twisted.python import threadable
from twisted.trial.unittest import FailTest, SynchronousTestCase
@skipIf(threadingSkip, 'Platform does not support threads')
def testThreadedSynchronization(self):
    o = TestObject()
    errors = []

    def callMethodLots():
        try:
            for i in range(1000):
                o.aMethod()
        except AssertionError as e:
            errors.append(str(e))
    threads = []
    for x in range(5):
        t = threading.Thread(target=callMethodLots)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise FailTest(errors)