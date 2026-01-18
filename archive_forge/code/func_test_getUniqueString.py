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
def test_getUniqueString(self):
    name_one = self.getUniqueString()
    self.assertEqual('%s-%d' % (self.id(), 1), name_one)
    name_two = self.getUniqueString()
    self.assertEqual('%s-%d' % (self.id(), 2), name_two)