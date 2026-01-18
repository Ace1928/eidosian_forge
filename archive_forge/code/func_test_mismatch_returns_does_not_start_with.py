import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_mismatch_returns_does_not_start_with(self):
    matcher = StartsWith('bar')
    self.assertIsInstance(matcher.match('foo'), DoesNotStartWith)