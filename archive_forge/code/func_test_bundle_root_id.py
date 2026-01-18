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
def test_bundle_root_id(self):
    self.tree1 = self.make_branch_and_tree('b1')
    self.b1 = self.tree1.branch
    self.tree1.commit('message', rev_id=b'revid1')
    bundle = self.get_valid_bundle(b'null:', b'revid1')
    tree = self.get_bundle_tree(bundle, b'revid1')
    root_revision = tree.get_file_revision('')
    self.assertEqual(b'revid1', root_revision)