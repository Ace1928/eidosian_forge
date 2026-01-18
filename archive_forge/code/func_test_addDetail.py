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
def test_addDetail(self):
    mycontent = self.get_content()
    self.addDetail('foo', mycontent)
    details = self.getDetails()
    self.assertEqual({'foo': mycontent}, details)