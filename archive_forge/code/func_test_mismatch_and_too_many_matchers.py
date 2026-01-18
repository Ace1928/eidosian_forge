import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_mismatch_and_too_many_matchers(self):
    self.assertMismatchWithDescriptionMatching([2, 3], MatchesSetwise(Equals(0), Equals(1), Equals(2)), MatchesRegex('.*There was 1 mismatch and 1 extra matcher: Equals\\([01]\\)', re.S))