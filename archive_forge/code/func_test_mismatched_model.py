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
def test_mismatched_model(self):
    """Try copying a bundle from knit2 to knit1"""
    format = bzrdir.BzrDirMetaFormat1()
    format.repository_format = knitrepo.RepositoryFormatKnit3()
    source = self.make_branch_and_tree('source', format=format)
    source.commit('one', rev_id=b'one-id')
    source.commit('two', rev_id=b'two-id')
    text = BytesIO()
    write_bundle(source.branch.repository, b'two-id', b'null:', text, format='0.9')
    text.seek(0)
    format = bzrdir.BzrDirMetaFormat1()
    format.repository_format = knitrepo.RepositoryFormatKnit1()
    target = self.make_branch('target', format=format)
    self.assertRaises(errors.IncompatibleRevision, install_bundle, target.repository, read_bundle(text))