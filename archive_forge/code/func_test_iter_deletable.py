import errno
import os
import shutil
import sys
from .. import tests, ui
from ..clean_tree import clean_tree, iter_deletables
from ..controldir import ControlDir
from ..osutils import supports_symlinks
from . import TestCaseInTempDir
def test_iter_deletable(self):
    """Files are selected for deletion appropriately"""
    os.mkdir('branch')
    tree = ControlDir.create_standalone_workingtree('branch')
    transport = tree.controldir.root_transport
    transport.put_bytes('.bzrignore', b'*~\n*.pyc\n.bzrignore\n')
    transport.put_bytes('file.BASE', b'contents')
    with tree.lock_write():
        self.assertEqual(len(list(iter_deletables(tree, unknown=True))), 1)
        transport.put_bytes('file', b'contents')
        transport.put_bytes('file~', b'contents')
        transport.put_bytes('file.pyc', b'contents')
        dels = sorted([r for a, r in iter_deletables(tree, unknown=True)])
        self.assertEqual(['file', 'file.BASE'], dels)
        dels = [r for a, r in iter_deletables(tree, detritus=True)]
        self.assertEqual(sorted(['file~', 'file.BASE']), dels)
        dels = [r for a, r in iter_deletables(tree, ignored=True)]
        self.assertEqual(sorted(['file~', 'file.pyc', '.bzrignore']), dels)
        dels = [r for a, r in iter_deletables(tree, unknown=False)]
        self.assertEqual([], dels)