import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def test_colo_checkout(self):
    source = self.make_branch_and_tree('source', format='development-colo')
    self.build_tree(['source/file1'])
    source.add('file1')
    source.commit('added file')
    target = source.controldir.sprout('file:second,branch=somebranch', create_tree_if_local=False)
    out, err = self.run_bzr('checkout file:,branch=somebranch .', working_dir='second')
    self.assertEqual(target.open_branch(name='somebranch').user_url, target.get_branch_reference(name=''))