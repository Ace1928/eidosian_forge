from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_default_description_is_mismatch(self):
    mismatch = Equals(3).match(2)
    e = MismatchError(2, Equals(3), mismatch)
    self.assertEqual(mismatch.describe(), str(e))