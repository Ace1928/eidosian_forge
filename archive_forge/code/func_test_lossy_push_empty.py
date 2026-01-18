from .. import (branch, controldir, errors, foreign, lockable_files, lockdir,
from .. import transport as _mod_transport
from ..bzr import branch as bzrbranch
from ..bzr import bzrdir, groupcompress_repo, vf_repository
from ..bzr.pack_repo import PackCommitBuilder
def test_lossy_push_empty(self):
    source_tree = self.make_branch_and_tree('source')
    target_tree = self.make_branch_and_tree('target', format=DummyForeignVcsDirFormat())
    pushresult = source_tree.branch.push(target_tree.branch, lossy=True)
    self.assertEqual(revision.NULL_REVISION, pushresult.old_revid)
    self.assertEqual(revision.NULL_REVISION, pushresult.new_revid)
    self.assertEqual({}, pushresult.revidmap)