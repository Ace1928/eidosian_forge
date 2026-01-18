import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_long_mixed_strings(self):
    mismatch = _BinaryMismatch(self._long_b, '!~', self._long_u)
    self.assertEqual(mismatch.describe(), '{}:\nreference = {}\nactual    = {}\n'.format('!~', text_repr(self._long_u, multiline=True), text_repr(self._long_b, multiline=True)))