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
def make_merged_branch(self):
    builder = self.make_branch_builder('source')
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'root-id', 'directory', None)), ('add', ('file', b'file-id', 'file', b'original content\n'))], revision_id=b'a@cset-0-1')
    builder.build_snapshot([b'a@cset-0-1'], [('modify', ('file', b'new-content\n'))], revision_id=b'a@cset-0-2a')
    builder.build_snapshot([b'a@cset-0-1'], [('add', ('other-file', b'file2-id', 'file', b'file2-content\n'))], revision_id=b'a@cset-0-2b')
    builder.build_snapshot([b'a@cset-0-2a', b'a@cset-0-2b'], [('add', ('other-file', b'file2-id', 'file', b'file2-content\n'))], revision_id=b'a@cset-0-3')
    builder.finish_series()
    self.b1 = builder.get_branch()
    self.b1.lock_read()
    self.addCleanup(self.b1.unlock)