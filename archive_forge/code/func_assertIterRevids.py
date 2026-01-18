from breezy import errors, revision, tests
from breezy.tests import per_branch
def assertIterRevids(self, expected, branch, *args, **kwargs):
    if kwargs.get('stop_revision_id') is not None:
        kwargs['stop_revision_id'] = self.revids[kwargs['stop_revision_id']]
    if kwargs.get('start_revision_id') is not None:
        kwargs['start_revision_id'] = self.revids[kwargs['start_revision_id']]
    revids = [revid for revid, depth, revno, eom in branch.iter_merge_sorted_revisions(*args, **kwargs)]
    self.assertEqual([self.revids[short] for short in expected], revids)