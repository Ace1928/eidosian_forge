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
def test_mismatched_bundle(self):
    format = bzrdir.BzrDirMetaFormat1()
    format.repository_format = knitrepo.RepositoryFormatKnit3()
    serializer = BundleSerializerV08('0.8')
    b = self.make_branch('.', format=format)
    self.assertRaises(errors.IncompatibleBundleFormat, serializer.write, b.repository, [], {}, BytesIO())