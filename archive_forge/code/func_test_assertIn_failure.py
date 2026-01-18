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
def test_assertIn_failure(self):
    self.assertFails('3 not in [0, 1, 2]', self.assertIn, 3, [0, 1, 2])
    self.assertFails('{!r} not in {!r}'.format('qux', 'foo bar baz'), self.assertIn, 'qux', 'foo bar baz')