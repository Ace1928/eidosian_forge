import testtools
from testtools import matchers as tt_matchers
from keystoneauth1.tests.unit import matchers as ks_matchers
def test_describe_difference(self):
    examples = self.describe_examples
    for difference, matchee, matcher in examples:
        mismatch = matcher.match(matchee)
        self.assertEqual(difference, mismatch.describe())