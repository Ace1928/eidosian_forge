from promise import Promise
from .utils import assert_exception
from threading import Event
def test_3_2_1_2():
    """
    That that the second argument to 'then' is ignored if it
    is not a function.
    """
    results = {}
    nonFunctions = [None, False, 5, {}, []]

    def testNonFunction(nonFunction):

        def foo(k, r):
            results[k] = r
        p1 = Promise.resolve('Error: ' + str(nonFunction))
        p2 = p1.then(lambda r: foo(str(nonFunction), r), nonFunction)
        p2._wait()
    for v in nonFunctions:
        testNonFunction(v)
    for v in nonFunctions:
        assert 'Error: ' + str(v) == results[str(v)]