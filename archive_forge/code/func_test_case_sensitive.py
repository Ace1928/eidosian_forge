import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
def test_case_sensitive(self):
    """If filesystem is case-sensitive, tree should report this.

        We check case-sensitivity by creating a file with a lowercase name,
        then testing whether it exists with an uppercase name.
        """
    self.build_tree(['filename'])
    case_sensitive = not features.CaseInsensitiveFilesystemFeature.available()
    tree = self.make_branch_and_tree('test')
    self.assertEqual(case_sensitive, tree.case_sensitive)
    if not isinstance(tree, InventoryWorkingTree):
        raise TestNotApplicable('get_format_string is only available on bzr working trees')
    t = tree.controldir.get_workingtree_transport(None)
    try:
        content = tree._format.get_format_string()
    except NotImplementedError:
        content = tree.controldir._format.get_format_string()
    t.put_bytes(tree._format.case_sensitive_filename, content)
    tree = tree.controldir.open_workingtree()
    self.assertFalse(tree.case_sensitive)