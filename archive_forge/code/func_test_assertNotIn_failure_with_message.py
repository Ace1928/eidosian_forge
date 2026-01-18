from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_assertNotIn_failure_with_message(self):
    self.assertFails('[1, 2, 3] matches Contains(3): foo bar', self.assertNotIn, 3, [1, 2, 3], 'foo bar')
    self.assertFails("'foo bar baz' matches Contains('foo'): foo bar", self.assertNotIn, 'foo', 'foo bar baz', 'foo bar')