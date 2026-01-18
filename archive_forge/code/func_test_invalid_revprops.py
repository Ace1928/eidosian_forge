from breezy.tests import TestNotApplicable
from breezy.tests.per_repository import TestCaseWithRepository
def test_invalid_revprops(self):
    """Invalid revision properties"""
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    if not b.repository._format.supports_custom_revision_properties:
        raise TestNotApplicable('format does not support custom revision properties')
    self.assertRaises(ValueError, wt.commit, message='invalid', revprops={'what a silly property': 'fine'})
    self.assertRaises(ValueError, wt.commit, message='invalid', revprops=dict(number=13))