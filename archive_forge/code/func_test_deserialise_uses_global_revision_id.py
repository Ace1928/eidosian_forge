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
def test_deserialise_uses_global_revision_id(self):
    """If it is set, then we re-use the global revision id"""
    repo = self.make_repository('.', format=controldir.format_registry.get('knit')())
    inv_xml = b'<inventory format="5" revision_id="other-rev-id">\n</inventory>\n'
    self.assertRaises(AssertionError, repo._deserialise_inventory, b'test-rev-id', [inv_xml])
    inv = repo._deserialise_inventory(b'other-rev-id', [inv_xml])
    self.assertEqual(b'other-rev-id', inv.root.revision)