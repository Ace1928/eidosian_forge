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
def build_test_bundle(self):
    wt = self.make_branch_and_tree('b1')
    self.build_tree(['b1/one'])
    wt.add('one')
    wt.commit('add one', rev_id=b'a@cset-0-1')
    self.build_tree(['b1/two'])
    wt.add('two')
    wt.commit('add two', rev_id=b'a@cset-0-2', revprops={'branch-nick': 'test'})
    bundle_txt = BytesIO()
    rev_ids = write_bundle(wt.branch.repository, b'a@cset-0-2', b'a@cset-0-1', bundle_txt, self.format)
    self.assertEqual({b'a@cset-0-2'}, set(rev_ids))
    bundle_txt.seek(0, 0)
    return bundle_txt