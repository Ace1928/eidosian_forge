from .. import errors, osutils
from .. import revision as _mod_revision
from ..branch import Branch
from ..bzr import bzrdir, knitrepo, versionedfile
from ..upgrade import Convert
from ..workingtree import WorkingTree
from . import TestCaseWithTransport
from .test_revision import make_branches
def test_fetch_with_deltas_no_delta_closure(self):
    tree = self.make_branch_and_tree('source', format='dirstate')
    target = self.make_repository('target', format='pack-0.92')
    self.build_tree(['source/file'])
    tree.set_root_id(b'root-id')
    tree.add('file', ids=b'file-id')
    tree.commit('one', rev_id=b'rev-one')
    source = tree.branch.repository
    source.texts = versionedfile.RecordingVersionedFilesDecorator(source.texts)
    source.signatures = versionedfile.RecordingVersionedFilesDecorator(source.signatures)
    source.revisions = versionedfile.RecordingVersionedFilesDecorator(source.revisions)
    source.inventories = versionedfile.RecordingVersionedFilesDecorator(source.inventories)
    self.assertTrue(target._format._fetch_uses_deltas)
    target.fetch(source, revision_id=b'rev-one')
    self.assertEqual(('get_record_stream', [(b'file-id', b'rev-one')], target._format._fetch_order, False), self.find_get_record_stream(source.texts.calls))
    self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, False), self.find_get_record_stream(source.inventories.calls, 2))
    self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, False), self.find_get_record_stream(source.revisions.calls))
    signature_calls = source.signatures.calls[-1:]
    self.assertEqual(('get_record_stream', [(b'rev-one',)], target._format._fetch_order, False), self.find_get_record_stream(signature_calls))