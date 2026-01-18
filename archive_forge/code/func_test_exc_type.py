import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._exception import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_exc_type(self):
    self.assertThat(lambda: 1 / 0, raises(ZeroDivisionError))