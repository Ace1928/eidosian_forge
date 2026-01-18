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
def test_addCleanup_called_in_reverse_order(self):
    log = []
    test = make_test_case(self.getUniqueString(), set_up=lambda _: log.append('setUp'), test_body=lambda _: log.append('runTest'), tear_down=lambda _: log.append('tearDown'), cleanups=[lambda _: log.append('first'), lambda _: log.append('second')])
    test.run()
    self.assertThat(log, Equals(['setUp', 'runTest', 'tearDown', 'second', 'first']))