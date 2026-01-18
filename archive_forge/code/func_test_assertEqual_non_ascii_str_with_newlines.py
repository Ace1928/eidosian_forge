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
def test_assertEqual_non_ascii_str_with_newlines(self):
    message = 'Be careful mixing unicode and bytes'
    a = 'a\n§\n'
    b = 'Just a longish string so the more verbose output form is used.'
    expected_error = '\n'.join(['!=:', "reference = '''\\", 'a', repr('§')[1:-1], "'''", f'actual    = {b!r}', ': ' + message])
    self.assertFails(expected_error, self.assertEqual, a, b, message)