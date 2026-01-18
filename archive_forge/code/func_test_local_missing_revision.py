from unittest import TestLoader
from .... import config, tests
from ....bzr.bzrdir import BzrDir
from ....tests import TestCaseInTempDir
from ..emailer import EmailSender
def test_local_missing_revision(self):
    master_tree = self.make_branch_and_tree('master')
    self.build_tree(['master/a'])
    master_tree.add('a')
    master_tree.commit('a')
    child_tree = master_tree.controldir.sprout('child').open_workingtree()
    child_tree.branch.bind(master_tree.branch)
    self.build_tree(['master/c'])
    master_tree.add(['c'])
    revision_id = master_tree.commit('c')
    self.assertFalse(child_tree.branch.repository.has_revision(revision_id))
    sender = EmailSender(master_tree.branch, revision_id, master_tree.branch.get_config(), local_branch=child_tree.branch)
    self.assertIs(master_tree.branch.repository, sender.repository)