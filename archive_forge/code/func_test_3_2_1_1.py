from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_1_1():
    """
    That that the first argument to 'then' is ignored if it
    is not a function.
    """
    results = {}
    nonFunctions = [None, False, 5, {}, []]

    def testNonFunction(nonFunction):

        def foo(k, r):
            results[k] = r
        p1 = Promise.reject(Exception('Error: ' + str(nonFunction)))
        p2 = p1.then(nonFunction, lambda r: foo(str(nonFunction), r))
        p2._wait()
    for v in nonFunctions:
        testNonFunction(v)
    for v in nonFunctions:
        assert_exception(results[str(v)], Exception, 'Error: ' + str(v))