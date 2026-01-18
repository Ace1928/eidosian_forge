import unittest as pyunit
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest
def test_assertFailure_masked(self):
    """
        A single wrong assertFailure should fail the whole test.
        """

    class ExampleFailure(Exception):
        pass

    class TC(unittest.TestCase):
        failureException = ExampleFailure

        def test_assertFailure(self):
            d = defer.maybeDeferred(lambda: 1 / 0)
            self.assertFailure(d, OverflowError)
            self.assertFailure(d, ZeroDivisionError)
            return d
    test = TC('test_assertFailure')
    result = pyunit.TestResult()
    test.run(result)
    self.assertEqual(1, len(result.failures))