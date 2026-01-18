import re
from testtools import TestCase
from testtools.compat import (
from testtools.matchers._basic import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_describe_non_ascii_unicode(self):
    string = 'A§'
    suffix = 'B§'
    mismatch = DoesNotEndWith(string, suffix)
    self.assertEqual('{} does not end with {}.'.format(text_repr(string), text_repr(suffix)), mismatch.describe())