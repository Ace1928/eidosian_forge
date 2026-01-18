import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_long_unicode_and_object(self):
    obj = object()
    mismatch = _BinaryMismatch(self._long_u, '!~', obj)
    self.assertEqual(mismatch.describe(), '{}:\nreference = {}\nactual    = {}\n'.format('!~', repr(obj), text_repr(self._long_u, multiline=True)))