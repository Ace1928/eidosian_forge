import errno
import os
import sys
import time
from io import BytesIO
from breezy.bzr.transform import resolve_checkout
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from ... import osutils, tests, trace, transform, urlutils
from ...bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ...errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ...osutils import file_kind, pathjoin
from ...transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ...transport import FileExists
from ...tree import TreeChange
from .. import TestSkipped, features
from ..features import HardlinkFeature, SymlinkFeature
def test_move_dangling_ie(self):
    create_tree, root = self.transform()
    root = create_tree.root
    create_tree.new_file('name1', root, [b'hello1'], b'name1')
    create_tree.apply()
    delete_contents, root = self.transform()
    file = delete_contents.trans_id_tree_path('name1')
    delete_contents.delete_contents(file)
    delete_contents.apply()
    move_id, root = self.transform()
    name1 = move_id.trans_id_tree_path('name1')
    newdir = move_id.new_directory('dir', root, b'newdir')
    move_id.adjust_path('name2', newdir, name1)
    move_id.apply()