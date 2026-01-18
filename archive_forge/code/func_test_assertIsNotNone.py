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
def test_assertIsNotNone(self):
    self.assertIsNotNone(0)
    self.assertIsNotNone('0')
    expected_error = 'None matches Is(None)'
    self.assertFails(expected_error, self.assertIsNotNone, None)