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
def test_filename_merge(self):
    root_id = generate_ids.gen_root_id()
    base = TransformGroup('BASE', root_id)
    this = TransformGroup('THIS', root_id)
    other = TransformGroup('OTHER', root_id)
    base_a, this_a, other_a = (t.tt.new_directory('a', t.root, b'a') for t in [base, this, other])
    base_b, this_b, other_b = (t.tt.new_directory('b', t.root, b'b') for t in [base, this, other])
    base.tt.new_directory('c', base_a, b'c')
    this.tt.new_directory('c1', this_a, b'c')
    other.tt.new_directory('c', other_b, b'c')
    base.tt.new_directory('d', base_a, b'd')
    this.tt.new_directory('d1', this_b, b'd')
    other.tt.new_directory('d', other_a, b'd')
    base.tt.new_directory('e', base_a, b'e')
    this.tt.new_directory('e', this_a, b'e')
    other.tt.new_directory('e1', other_b, b'e')
    base.tt.new_directory('f', base_a, b'f')
    this.tt.new_directory('f1', this_b, b'f')
    other.tt.new_directory('f1', other_b, b'f')
    for tg in [this, base, other]:
        tg.tt.apply()
    Merge3Merger(this.wt, this.wt, base.wt, other.wt)
    self.assertEqual(this.wt.id2path(b'c'), pathjoin('b/c1'))
    self.assertEqual(this.wt.id2path(b'd'), pathjoin('b/d1'))
    self.assertEqual(this.wt.id2path(b'e'), pathjoin('b/e1'))
    self.assertEqual(this.wt.id2path(b'f'), pathjoin('b/f1'))