import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_with_symlinks(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    os.mkdir('a')
    a = ControlDir.create_standalone_workingtree('a')
    os.mkdir('a/foo')
    with open('a/foo/bar', 'wb') as f:
        f.write(b'contents')
    os.symlink('a/foo/bar', 'a/foo/baz')
    a.add(['foo', 'foo/bar', 'foo/baz'])
    a.commit('initial commit')
    b = ControlDir.create_standalone_workingtree('b')
    basis = a.basis_tree()
    basis.lock_read()
    self.addCleanup(basis.unlock)
    build_tree(basis, b)
    self.assertIs(os.path.isdir('b/foo'), True)
    with open('b/foo/bar', 'rb') as f:
        self.assertEqual(f.read(), b'contents')
    self.assertEqual(os.readlink('b/foo/baz'), 'a/foo/bar')