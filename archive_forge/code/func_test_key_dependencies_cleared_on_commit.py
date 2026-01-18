from stat import S_ISDIR
from ... import controldir, errors, gpg, osutils, repository
from ... import revision as _mod_revision
from ... import tests, transport, ui
from ...tests import TestCaseWithTransport, TestNotApplicable, test_server
from ...transport import memory
from .. import inventory
from ..btree_index import BTreeGraphIndex
from ..groupcompress_repo import RepositoryFormat2a
from ..index import GraphIndex
from ..smart import client
def test_key_dependencies_cleared_on_commit(self):
    source_repo, target_repo = self.create_source_and_target()
    target_repo.start_write_group()
    try:
        for vf_name in ['texts', 'chk_bytes', 'inventories']:
            source_vf = getattr(source_repo, vf_name, None)
            if source_vf is None:
                continue
            target_vf = getattr(target_repo, vf_name)
            stream = source_vf.get_record_stream(source_vf.keys(), 'unordered', True)
            target_vf.insert_record_stream(stream)
        stream = source_repo.revisions.get_record_stream([(b'B-id',)], 'unordered', True)
        target_repo.revisions.insert_record_stream(stream)
        key_refs = target_repo.revisions._index._key_dependencies
        self.assertEqual([(b'B-id',)], sorted(key_refs.get_referrers()))
    finally:
        target_repo.commit_write_group()
    self.assertEqual([], sorted(key_refs.get_referrers()))