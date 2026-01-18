from promise import Promise
from .utils import assert_exception
from threading import Event
def testNonFunction(nonFunction):

    def foo(k, r):
        results[k] = r
    p1 = Promise.resolve('Error: ' + str(nonFunction))
    p2 = p1.then(lambda r: foo(str(nonFunction), r), nonFunction)
    p2._wait()