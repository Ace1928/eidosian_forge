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
def test_bundle(self):
    self.tree1 = self.make_branch_and_tree('b1')
    self.b1 = self.tree1.branch
    self.build_tree_contents([('b1/one', b'one\n')])
    self.tree1.add('one', ids=b'one-id')
    self.tree1.set_root_id(b'root-id')
    self.tree1.commit('add one', rev_id=b'a@cset-0-1')
    bundle = self.get_valid_bundle(b'null:', b'a@cset-0-1')
    self.build_tree(['b1/with space.txt', 'b1/dir/', 'b1/dir/filein subdir.c', 'b1/dir/WithCaps.txt', 'b1/dir/ pre space', 'b1/sub/', 'b1/sub/sub/', 'b1/sub/sub/nonempty.txt'])
    self.build_tree_contents([('b1/sub/sub/emptyfile.txt', b''), ('b1/dir/nolastnewline.txt', b'bloop')])
    tt = self.tree1.transform()
    tt.new_file('executable', tt.root, [b'#!/bin/sh\n'], b'exe-1', True)
    tt.apply()
    self.tree1.add('with space.txt', ids=b'withspace-id')
    self.tree1.add(['dir', 'dir/filein subdir.c', 'dir/WithCaps.txt', 'dir/ pre space', 'dir/nolastnewline.txt', 'sub', 'sub/sub', 'sub/sub/nonempty.txt', 'sub/sub/emptyfile.txt'])
    self.tree1.commit('add whitespace', rev_id=b'a@cset-0-2')
    bundle = self.get_valid_bundle(b'a@cset-0-1', b'a@cset-0-2')
    bundle = self.get_valid_bundle(b'null:', b'a@cset-0-2')
    self.tree1.remove(['sub/sub/nonempty.txt', 'sub/sub/emptyfile.txt', 'sub/sub'])
    tt = self.tree1.transform()
    trans_id = tt.trans_id_tree_path('executable')
    tt.set_executability(False, trans_id)
    tt.apply()
    self.tree1.commit('removed', rev_id=b'a@cset-0-3')
    bundle = self.get_valid_bundle(b'a@cset-0-2', b'a@cset-0-3')
    self.assertRaises((errors.TestamentMismatch, errors.VersionedFileInvalidChecksum, errors.BadBundle), self.get_invalid_bundle, b'a@cset-0-2', b'a@cset-0-3')
    bundle = self.get_valid_bundle(b'null:', b'a@cset-0-3')
    self.tree1.rename_one('dir', 'sub/dir')
    self.tree1.commit('rename dir', rev_id=b'a@cset-0-4')
    bundle = self.get_valid_bundle(b'a@cset-0-3', b'a@cset-0-4')
    bundle = self.get_valid_bundle(b'null:', b'a@cset-0-4')
    with open('b1/sub/dir/WithCaps.txt', 'ab') as f:
        f.write(b'\nAdding some text\n')
    with open('b1/sub/dir/ pre space', 'ab') as f:
        f.write(b'\r\nAdding some\r\nDOS format lines\r\n')
    with open('b1/sub/dir/nolastnewline.txt', 'ab') as f:
        f.write(b'\n')
    self.tree1.rename_one('sub/dir/ pre space', 'sub/ start space')
    self.tree1.commit('Modified files', rev_id=b'a@cset-0-5')
    bundle = self.get_valid_bundle(b'a@cset-0-4', b'a@cset-0-5')
    self.tree1.rename_one('sub/dir/WithCaps.txt', 'temp')
    self.tree1.rename_one('with space.txt', 'WithCaps.txt')
    self.tree1.rename_one('temp', 'with space.txt')
    self.tree1.commit('swap filenames', rev_id=b'a@cset-0-6', verbose=False)
    bundle = self.get_valid_bundle(b'a@cset-0-5', b'a@cset-0-6')
    other = self.get_checkout(b'a@cset-0-5')
    tree1_inv = get_inventory_text(self.tree1.branch.repository, b'a@cset-0-5')
    tree2_inv = get_inventory_text(other.branch.repository, b'a@cset-0-5')
    self.assertEqualDiff(tree1_inv, tree2_inv)
    other.rename_one('sub/dir/nolastnewline.txt', 'sub/nolastnewline.txt')
    other.commit('rename file', rev_id=b'a@cset-0-6b')
    self.tree1.merge_from_branch(other.branch)
    self.tree1.commit('Merge', rev_id=b'a@cset-0-7', verbose=False)
    bundle = self.get_valid_bundle(b'a@cset-0-6', b'a@cset-0-7')