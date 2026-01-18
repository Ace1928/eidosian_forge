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
def test_multipleCleanupErrorsReported(self):
    test = make_test_case(self.getUniqueString(), cleanups=[lambda _: 1 / 0, lambda _: 1 / 0])
    log = []
    test.run(ExtendedTestResult(log))
    self.assertThat(log, MatchesEvents(('startTest', test), ('addError', test, {'traceback': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError'])), 'traceback-1': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError']))}), ('stopTest', test)))