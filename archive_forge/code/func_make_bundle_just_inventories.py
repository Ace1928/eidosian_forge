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
def make_bundle_just_inventories(self, base_revision_id, target_revision_id, revision_ids):
    sio = BytesIO()
    writer = v4.BundleWriteOperation(base_revision_id, target_revision_id, self.b1.repository, sio)
    writer.bundle.begin()
    writer._add_inventory_mpdiffs_from_serializer(revision_ids)
    writer.bundle.end()
    sio.seek(0)
    return sio