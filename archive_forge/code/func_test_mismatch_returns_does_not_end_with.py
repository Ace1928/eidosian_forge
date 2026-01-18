import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_mismatch_returns_does_not_end_with(self):
    matcher = EndsWith('bar')
    self.assertIsInstance(matcher.match('foo'), DoesNotEndWith)