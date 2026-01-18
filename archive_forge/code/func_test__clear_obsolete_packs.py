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
def test__clear_obsolete_packs(self):
    packs = self.get_packs()
    obsolete_pack_trans = packs.transport.clone('obsolete_packs')
    obsolete_pack_trans.put_bytes('a-pack.pack', b'content\n')
    obsolete_pack_trans.put_bytes('a-pack.rix', b'content\n')
    obsolete_pack_trans.put_bytes('a-pack.iix', b'content\n')
    obsolete_pack_trans.put_bytes('another-pack.pack', b'foo\n')
    obsolete_pack_trans.put_bytes('not-a-pack.rix', b'foo\n')
    res = packs._clear_obsolete_packs()
    self.assertEqual(['a-pack', 'another-pack'], sorted(res))
    self.assertEqual([], obsolete_pack_trans.list_dir('.'))