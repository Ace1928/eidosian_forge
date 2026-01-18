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
def test_both_rename(self):
    create_tree, root = self.transform()
    newdir = create_tree.new_directory('selftest', root, b'selftest-id')
    create_tree.new_file('blackbox.py', newdir, [b'hello1'], b'blackbox-id')
    create_tree.apply()
    mangle_tree, root = self.transform()
    selftest = mangle_tree.trans_id_tree_path('selftest')
    blackbox = mangle_tree.trans_id_tree_path('selftest/blackbox.py')
    mangle_tree.adjust_path('test', root, selftest)
    mangle_tree.adjust_path('test_too_much', root, selftest)
    mangle_tree.set_executability(True, blackbox)
    mangle_tree.apply()