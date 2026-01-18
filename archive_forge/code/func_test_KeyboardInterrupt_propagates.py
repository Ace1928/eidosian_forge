import sys
from testtools import TestCase
from testtools.matchers import (
from testtools.matchers._exception import (
from testtools.tests.helpers import FullStackRunTest
from testtools.tests.matchers.helpers import TestMatchersInterface
def test_KeyboardInterrupt_propagates(self):
    match_keyb = Raises(MatchesException(KeyboardInterrupt))

    def raise_keyb_from_match():
        matcher = Raises()
        matcher.match(self.raiser)
    self.assertThat(raise_keyb_from_match, match_keyb)