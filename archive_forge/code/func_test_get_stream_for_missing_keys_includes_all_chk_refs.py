from stat import S_ISDIR
import breezy
from breezy import controldir, errors, osutils, repository
from breezy import revision as _mod_revision
from breezy import tests, transport, upgrade, workingtree
from breezy.bzr import (btree_index, bzrdir, groupcompress_repo, inventory,
from breezy.bzr import repository as bzrrepository
from breezy.bzr import versionedfile, vf_repository, vf_search
from breezy.bzr.btree_index import BTreeBuilder, BTreeGraphIndex
from breezy.bzr.index import GraphIndex
from breezy.errors import UnknownFormatError
from breezy.repository import RepositoryFormat
from breezy.tests import TestCase, TestCaseWithTransport
def test_get_stream_for_missing_keys_includes_all_chk_refs(self):
    source_builder = self.make_branch_builder('source', format='2a')
    entries = [('add', ('', b'a-root-id', 'directory', None))]
    for i in 'abcdefghijklmnopqrstuvwxyz123456789':
        for j in 'abcdefghijklmnopqrstuvwxyz123456789':
            fname = i + j
            fid = fname.encode('utf-8') + b'-id'
            content = b'content for %s\n' % (fname.encode('utf-8'),)
            entries.append(('add', (fname, fid, 'file', content)))
    source_builder.start_series()
    source_builder.build_snapshot(None, entries, revision_id=b'rev-1')
    source_builder.build_snapshot([b'rev-1'], [('modify', ('aa', b'new content for aa-id\n')), ('modify', ('cc', b'new content for cc-id\n')), ('modify', ('zz', b'new content for zz-id\n'))], revision_id=b'rev-2')
    source_builder.finish_series()
    source_branch = source_builder.get_branch()
    source_branch.lock_read()
    self.addCleanup(source_branch.unlock)
    target = self.make_repository('target', format='2a')
    source = source_branch.repository._get_source(target._format)
    self.assertIsInstance(source, groupcompress_repo.GroupCHKStreamSource)
    search = vf_search.SearchResult({b'rev-2'}, {b'rev-1'}, 1, {b'rev-2'})
    simple_chk_records = set()
    for vf_name, substream in source.get_stream(search):
        if vf_name == 'chk_bytes':
            for record in substream:
                simple_chk_records.add(record.key)
        else:
            for _ in substream:
                continue
    self.assertEqual({(b'sha1:91481f539e802c76542ea5e4c83ad416bf219f73',), (b'sha1:4ff91971043668583985aec83f4f0ab10a907d3f',), (b'sha1:81e7324507c5ca132eedaf2d8414ee4bb2226187',), (b'sha1:b101b7da280596c71a4540e9a1eeba8045985ee0',)}, set(simple_chk_records))
    missing = [('inventories', b'rev-2')]
    full_chk_records = set()
    for vf_name, substream in source.get_stream_for_missing_keys(missing):
        if vf_name == 'inventories':
            for record in substream:
                self.assertEqual((b'rev-2',), record.key)
        elif vf_name == 'chk_bytes':
            for record in substream:
                full_chk_records.add(record.key)
        else:
            self.fail('Should not be getting a stream of {}'.format(vf_name))
    self.assertEqual(257, len(full_chk_records))
    self.assertSubset(simple_chk_records, full_chk_records)