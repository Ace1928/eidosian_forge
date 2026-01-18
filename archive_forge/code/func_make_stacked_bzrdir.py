from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def make_stacked_bzrdir(self, in_directory=None):
    """Create a stacked branch and return its bzrdir.

        :param in_directory: If not None, create a directory of this
            name and create the stacking and stacked-on bzrdirs in
            this directory.
        """
    if in_directory is not None:
        self.get_transport().mkdir(in_directory)
        prefix = in_directory + '/'
    else:
        prefix = ''
    tree = self.make_branch_and_tree(prefix + 'stacked-on')
    tree.commit('Added foo')
    stacked_bzrdir = tree.branch.controldir.sprout(self.get_url(prefix + 'stacked'), tree.branch.last_revision(), stacked=True)
    return stacked_bzrdir