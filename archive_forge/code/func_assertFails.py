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
def assertFails(self, message, function, *args, **kwargs):
    """Assert that function raises a failure with the given message."""
    failure = self.assertRaises(self.failureException, function, *args, **kwargs)
    self.assertThat(failure, DocTestMatches(message, ELLIPSIS))