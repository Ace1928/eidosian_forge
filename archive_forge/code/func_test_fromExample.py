import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_fromExample(self):
    self.assertThat(self.SimpleClass(1, 2), MatchesStructure.fromExample(self.SimpleClass(1, 3), 'x'))