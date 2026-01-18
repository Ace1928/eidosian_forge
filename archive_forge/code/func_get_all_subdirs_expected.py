import os
from breezy import tests
from breezy.mutabletree import MutableTree
from breezy.osutils import supports_symlinks
from breezy.tests.per_tree import TestCaseWithTree
def get_all_subdirs_expected(self, tree, symlinks):
    empty_dirs_present = tree.has_versioned_directories() or isinstance(tree, MutableTree)
    empty_dirs_are_versioned = tree.has_versioned_directories()
    dirblocks = {}
    dirblocks[''] = [('0file', '0file', 'file', None, 'file'), ('1top-dir', '1top-dir', 'directory', None, 'directory'), ('2utfሴfile', '2utfሴfile', 'file', None, 'file')]
    dirblocks['1top-dir'] = [('1top-dir/0file-in-1topdir', '0file-in-1topdir', 'file', None, 'file')]
    if empty_dirs_present:
        dirblocks['1top-dir'].append(('1top-dir/1dir-in-1topdir', '1dir-in-1topdir', 'directory', None if empty_dirs_are_versioned else os.stat(tree.abspath('1top-dir/1dir-in-1topdir')), 'directory' if empty_dirs_are_versioned else None))
        dirblocks['1top-dir/1dir-in-1topdir'] = []
    if symlinks:
        dirblocks[''].append(('symlink', 'symlink', 'symlink', None, 'symlink'))
    return [(path, list(sorted(entries))) for path, entries in sorted(dirblocks.items())]