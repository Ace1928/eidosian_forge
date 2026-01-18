from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_fetch_revisions_with_deltas_into_pack(self):
    tree = self.make_branch_and_tree('source', format='dirstate')
    target = self.make_repository('target', format='pack-0.92')
    self.build_tree(['source/file'])
    tree.set_root_id(b'root-id')
    tree.add('file', ids=b'file-id')
    tree.commit('one', rev_id=b'rev-one')
    tree.branch.repository.revisions._max_delta_chain = 200
    tree.commit('two', rev_id=b'rev-two')
    source = tree.branch.repository
    source.lock_read()
    self.addCleanup(source.unlock)
    record = next(source.revisions.get_record_stream([(b'rev-two',)], 'unordered', False))
    self.assertEqual('knit-delta-gz', record.storage_kind)
    target.fetch(tree.branch.repository, revision_id=b'rev-two')
    target.lock_read()
    self.addCleanup(target.unlock)
    record = next(target.revisions.get_record_stream([(b'rev-two',)], 'unordered', False))
    self.assertEqual('knit-ft-gz', record.storage_kind)