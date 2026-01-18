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
def test_before_after_hooks(self):
    case = DecorateTestCaseResult(PlaceHolder('foo'), self.make_result, before_run=lambda result: self.log.append('before'), after_run=lambda result: self.log.append('after'))
    case.run(None)
    case(None)
    self.assertEqual([('result', None), 'before', ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set()), 'after', ('result', None), 'before', ('tags', set(), set()), ('startTest', case.decorated), ('addSuccess', case.decorated), ('stopTest', case.decorated), ('tags', set(), set()), 'after'], self.log)