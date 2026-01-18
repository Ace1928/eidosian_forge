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
def test_assertIs_fails(self):
    self.assertFails('42 is not None', self.assertIs, None, 42)
    self.assertFails('[42] is not [42]', self.assertIs, [42], [42])