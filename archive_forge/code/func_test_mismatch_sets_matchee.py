import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_mismatch_sets_matchee(self):
    matcher = EndsWith('bar')
    mismatch = matcher.match('foo')
    self.assertEqual('foo', mismatch.matchee)