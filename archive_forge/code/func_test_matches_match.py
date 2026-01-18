import testtools
from testtools import matchers as tt_matchers
from keystoneauth1.tests.unit import matchers as ks_matchers
def test_matches_match(self):
    matcher = self.matches_matcher
    matches = self.matches_matches
    mismatches = self.matches_mismatches
    for candidate in matches:
        self.assertIsNone(matcher.match(candidate))
    for candidate in mismatches:
        mismatch = matcher.match(candidate)
        self.assertIsNotNone(mismatch)
        self.assertIsNotNone(getattr(mismatch, 'describe', None))