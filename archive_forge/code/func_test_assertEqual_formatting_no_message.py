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
def test_assertEqual_formatting_no_message(self):
    a = 'cat'
    b = 'dog'
    expected_error = "'cat' != 'dog'"
    self.assertFails(expected_error, self.assertEqual, a, b)
    self.assertFails(expected_error, self.assertEquals, a, b)
    self.assertFails(expected_error, self.failUnlessEqual, a, b)