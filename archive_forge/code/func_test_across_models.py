import bz2
import os
import sys
import tempfile
from io import BytesIO
from ... import diff, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...tests import features, test_commit
from ...tree import InterTree
from .. import bzrdir, inventory, knitrepo
from ..bundle.apply_bundle import install_bundle, merge_bundle
from ..bundle.bundle_data import BundleTree
from ..bundle.serializer import read_bundle, v4, v09, write_bundle
from ..bundle.serializer.v4 import BundleSerializerV4
from ..bundle.serializer.v08 import BundleSerializerV08
from ..bundle.serializer.v09 import BundleSerializerV09
from ..inventorytree import InventoryTree
def test_across_models(self):
    repo = self.make_repo_with_installed_revisions()
    inv = repo.get_inventory(b'rev2')
    self.assertEqual(b'rev2', inv.root.revision)
    root_id = inv.root.file_id
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertEqual({(root_id, b'rev1'): (), (root_id, b'rev2'): ((root_id, b'rev1'),)}, repo.texts.get_parent_map([(root_id, b'rev1'), (root_id, b'rev2')]))