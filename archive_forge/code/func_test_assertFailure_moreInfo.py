import unittest as pyunit
from twisted.internet import defer
from twisted.python import failure
from twisted.trial import unittest
def test_assertFailure_moreInfo(self):
    """
        In the case of assertFailure failing, check that we get lots of
        information about the exception that was raised.
        """
    try:
        1 / 0
    except ZeroDivisionError:
        f = failure.Failure()
        d = defer.fail(f)
    d = self.assertFailure(d, RuntimeError)
    d.addErrback(self._checkInfo, f)
    return d