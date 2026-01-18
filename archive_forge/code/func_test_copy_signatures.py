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
def test_copy_signatures(self):
    tree_a = self.make_branch_and_tree('tree_a')
    import breezy.commit as commit
    import breezy.gpg
    oldstrategy = breezy.gpg.GPGStrategy
    branch = tree_a.branch
    repo_a = branch.repository
    tree_a.commit('base', allow_pointless=True, rev_id=b'A')
    self.assertFalse(branch.repository.has_signature_for_revision_id(b'A'))
    try:
        from ..testament import Testament
        breezy.gpg.GPGStrategy = breezy.gpg.LoopbackGPGStrategy
        new_config = test_commit.MustSignConfig()
        commit.Commit(config_stack=new_config).commit(message='base', allow_pointless=True, rev_id=b'B', working_tree=tree_a)

        def sign(text):
            return breezy.gpg.LoopbackGPGStrategy(None).sign(text)
        self.assertTrue(repo_a.has_signature_for_revision_id(b'B'))
    finally:
        breezy.gpg.GPGStrategy = oldstrategy
    tree_b = self.make_branch_and_tree('tree_b')
    repo_b = tree_b.branch.repository
    s = BytesIO()
    serializer = BundleSerializerV4('4')
    with tree_a.lock_read():
        serializer.write_bundle(tree_a.branch.repository, b'B', b'null:', s)
    s.seek(0)
    install_bundle(repo_b, serializer.read(s))
    self.assertTrue(repo_b.has_signature_for_revision_id(b'B'))
    self.assertEqual(repo_b.get_signature_text(b'B'), repo_a.get_signature_text(b'B'))
    s.seek(0)
    install_bundle(repo_b, serializer.read(s))