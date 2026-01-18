import doctest
import io
import re
import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._datastructures import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def assertMismatchWithDescriptionMatching(self, value, matcher, description_matcher):
    mismatch = matcher.match(value)
    if mismatch is None:
        self.fail(f'{matcher} matched {value}')
    actual_description = mismatch.describe()
    self.assertThat(actual_description, Annotate(f'{matcher} matching {value}', description_matcher))