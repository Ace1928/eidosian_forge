from testtools import (
from testtools.compat import (
from testtools.matchers import (
from testtools.matchers._impl import (
from testtools.tests.helpers import FullStackRunTest
def test_default_description_unicode(self):
    matchee = '§'
    matcher = Equals('a')
    mismatch = matcher.match(matchee)
    e = MismatchError(matchee, matcher, mismatch)
    self.assertEqual(mismatch.describe(), str(e))