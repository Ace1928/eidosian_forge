from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def make_branch_with_revision_tuple(self, relpath, count):
    builder = self.make_branch_builder(relpath)
    builder.start_series()
    revids = [builder.build_commit() for i in range(count)]
    builder.finish_series()
    return (builder.get_branch(), revids)