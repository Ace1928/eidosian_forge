from breezy import branch
from breezy.errors import NoRoundtrippingSupport
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import (StubMatchingInter, StubWithFormat,
def test_contract_convenience_method(self):
    self.tree1 = self.make_from_branch_and_tree('tree1')
    rev1 = self.tree1.commit('one')
    branch2 = self.make_to_branch('tree2')
    try:
        branch2.repository.fetch(self.tree1.branch.repository)
    except NoRoundtrippingSupport:
        raise TestNotApplicable('lossless cross-vcs fetch from %r to %r unsupported' % (self.tree1.branch, branch2))
    self.tree1.branch.copy_content_into(branch2, revision_id=rev1)