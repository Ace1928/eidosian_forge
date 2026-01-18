from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def make_branch_with_revisions(self, relpath, revisions):
    builder = self.make_branch_builder(relpath)
    builder.start_series()
    for revid in revisions:
        builder.build_commit(rev_id=revid)
    builder.finish_series()
    return builder.get_branch()