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
def test_concurrent_writer_second_preserves_dropping_a_pack(self):
    format = self.get_format()
    self.make_repository('.', shared=True, format=format)
    r1 = repository.Repository.open('.')
    r2 = repository.Repository.open('.')
    with r1.lock_write():
        with repository.WriteGroup(r1):
            self._add_text(r1, b'fileidr1')
        r1._pack_collection.ensure_loaded()
        name_to_drop = r1._pack_collection.all_packs()[0].name
    with r1.lock_write():
        list(r1.all_revision_ids())
        with r2.lock_write():
            list(r2.all_revision_ids())
            r1._pack_collection.ensure_loaded()
            try:
                r2.start_write_group()
                try:
                    r1._pack_collection._remove_pack_from_memory(r1._pack_collection.get_pack_by_name(name_to_drop))
                    self._add_text(r2, b'fileidr2')
                except:
                    r2.abort_write_group()
                    raise
            except:
                r1._pack_collection.reset()
                raise
            try:
                r1._pack_collection._save_pack_names()
                r1._pack_collection.reset()
            except:
                r2.abort_write_group()
                raise
            try:
                r2.commit_write_group()
            except:
                r2.abort_write_group()
                raise
            r1._pack_collection.ensure_loaded()
            r2._pack_collection.ensure_loaded()
            self.assertEqual(r1._pack_collection.names(), r2._pack_collection.names())
            self.assertEqual(1, len(r1._pack_collection.names()))
            self.assertFalse(name_to_drop in r1._pack_collection.names())