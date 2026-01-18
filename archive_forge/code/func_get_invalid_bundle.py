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
def get_invalid_bundle(self, base_rev_id, rev_id):
    """Create a bundle from base_rev_id -> rev_id in built-in branch.
        Munge the text so that it's invalid.

        :return: The in-memory bundle
        """
    from ..bundle import serializer
    bundle_txt, rev_ids = self.create_bundle_text(base_rev_id, rev_id)
    new_text = self.get_raw(BytesIO(b''.join(bundle_txt)))
    self.assertContainsRe(new_text, b'(?m)B244\n\ni 1\n<inventory')
    new_text = new_text.replace(b'<file file_id="exe-1"', b'<file executable="y" file_id="exe-1"')
    new_text = new_text.replace(b'B244', b'B259')
    bundle_txt = BytesIO()
    bundle_txt.write(serializer._get_bundle_header('4'))
    bundle_txt.write(b'\n')
    bundle_txt.write(bz2.compress(new_text))
    bundle_txt.seek(0)
    bundle = read_bundle(bundle_txt)
    self.valid_apply_bundle(base_rev_id, bundle)
    return bundle