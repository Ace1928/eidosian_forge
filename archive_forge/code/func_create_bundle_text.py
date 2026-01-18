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
def create_bundle_text(self, base_rev_id, rev_id):
    bundle_txt = BytesIO()
    rev_ids = write_bundle(self.b1.repository, rev_id, base_rev_id, bundle_txt, format=self.format)
    bundle_txt.seek(0)
    self.assertEqual(bundle_txt.readline(), b'# Bazaar revision bundle v%s\n' % self.format.encode('ascii'))
    self.assertEqual(bundle_txt.readline(), b'#\n')
    rev = self.b1.repository.get_revision(rev_id)
    bundle_txt.seek(0)
    return (bundle_txt, rev_ids)