import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_long_bytes(self):
    one_line_b = self._long_b.replace(_b('\n'), _b(' '))
    mismatch = _BinaryMismatch(one_line_b, '!~', self._long_b)
    self.assertEqual(mismatch.describe(), '{}:\nreference = {}\nactual    = {}\n'.format('!~', text_repr(self._long_b, multiline=True), text_repr(one_line_b)))