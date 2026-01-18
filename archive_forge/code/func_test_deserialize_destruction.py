import codecs
import errno
import os
import sys
import time
from io import BytesIO, StringIO
import fastbencode as bencode
from .. import filters, osutils
from .. import revision as _mod_revision
from .. import rules, tests, trace, transform, urlutils
from ..bzr import generate_ids
from ..bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ..controldir import ControlDir
from ..diff import show_diff_trees
from ..errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ..merge import Merge3Merger, Merger
from ..mutabletree import MutableTree
from ..osutils import file_kind, pathjoin
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..transport import FileExists
from . import TestCaseInTempDir, TestSkipped, features
from .features import HardlinkFeature, SymlinkFeature
def test_deserialize_destruction(self):
    tt = self.make_destruction_preview()
    tt.deserialize(iter(self.destruction_records()))
    self.assertEqual({'fooሴ': 'new-1', 'bar': 'new-2', '': tt.root}, tt._tree_path_ids)
    self.assertEqual({'new-1': 'fooሴ', 'new-2': 'bar', tt.root: ''}, tt._tree_id_paths)
    self.assertEqual({'new-1'}, tt._removed_id)
    self.assertEqual({'new-2'}, tt._removed_contents)