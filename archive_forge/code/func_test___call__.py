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
def test___call__(self):
    case = DecorateTestCaseResult(PlaceHolder('foo'), self.make_result)
    case(None)
    case('something')
    self.assertEqual([('result', None), ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set()), ('result', 'something'), ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set())], self.log)