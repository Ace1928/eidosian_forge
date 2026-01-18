from breezy.tests.per_controldir import TestCaseWithControlDir
from ...controldir import NoColocatedBranchSupport
from ...errors import LossyPushToSameVCS, NoSuchRevision, TagsNotSupported
from ...revision import NULL_REVISION
from .. import TestNotApplicable
def create_simple_tree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a'])
    tree.add(['a'])
    rev_1 = tree.commit('one')
    return (tree, rev_1)