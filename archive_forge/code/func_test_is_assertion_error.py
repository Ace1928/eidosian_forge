from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_is_assertion_error(self):

    def raise_mismatch_error():
        raise MismatchError(2, Equals(3), Equals(3).match(2))
    self.assertRaises(AssertionError, raise_mismatch_error)