import contextlib
from breezy import errors, tests, transform, transport
from breezy.bzr.workingtree_4 import (DirStateRevisionTree, WorkingTreeFormat4,
from breezy.git.tree import GitRevisionTree
from breezy.git.workingtree import GitWorkingTreeFormat
from breezy.revisiontree import RevisionTree
from breezy.tests import features
from breezy.tests.per_controldir.test_controldir import TestCaseWithControlDir
from breezy.tests.per_workingtree import make_scenario as wt_make_scenario
from breezy.tests.per_workingtree import make_scenarios as wt_make_scenarios
from breezy.workingtree import format_registry
def get_tree_with_subdirs_and_all_supported_content_types(self, symlinks):
    """Return a test tree with subdirs and all supported content types.
        Some content types may not be created on some platforms
        (like symlinks on native win32)

        :param  symlinks:   control is symlink should be created in the tree.
                            Note: if you wish to automatically set this
                            parameters depending on underlying system,
                            please use value returned
                            by breezy.osutils.supports_symlinks() function.

        The returned tree has the following inventory:
            ['',
             '0file',
             '1top-dir',
             u'2utfሴfile',
             'symlink',            # only if symlinks arg is True
             '1top-dir/0file-in-1topdir',
             '1top-dir/1dir-in-1topdir']
        where each component has the type of its name -
        i.e. '1file..' is afile.

        note that the order of the paths and fileids is deliberately
        mismatched to ensure that the result order is path based.
        """
    self.requireFeature(features.UnicodeFilenameFeature)
    tree = self.make_branch_and_tree('.')
    paths = ['0file', '1top-dir/', '2utfሴfile', '1top-dir/0file-in-1topdir', '1top-dir/1dir-in-1topdir/']
    self.build_tree(paths)
    tree.add(paths)
    tt = tree.transform()
    if symlinks:
        root_transaction_id = tt.trans_id_tree_path('')
        tt.new_symlink('symlink', root_transaction_id, 'link-target', b'symlink')
    tt.apply()
    return self.workingtree_to_test_tree(tree)