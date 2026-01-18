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
def test_assertThat_output(self):
    matchee = 'foo'
    matcher = Equals('bar')
    expected = matcher.match(matchee).describe()
    self.assertFails(expected, self.assertThat, matchee, matcher)