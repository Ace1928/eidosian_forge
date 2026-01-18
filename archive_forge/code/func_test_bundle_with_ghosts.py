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
def test_bundle_with_ghosts(self):
    tree = self.make_branch_and_tree('tree')
    self.b1 = tree.branch
    self.build_tree_contents([('tree/file', b'content1')])
    tree.add(['file'])
    tree.commit('rev1')
    self.build_tree_contents([('tree/file', b'content2')])
    tree.add_parent_tree_id(b'ghost')
    tree.commit('rev2', rev_id=b'rev2')
    bundle = self.get_valid_bundle(b'null:', b'rev2')