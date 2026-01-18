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
def test_multiple_attributes(self):
    case = Attributes('many')
    self.assertEqual('testtools.tests.test_testcase.Attributes.many[bar,foo,quux]', case.id())