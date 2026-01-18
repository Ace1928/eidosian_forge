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
def test_filename_merge_conflicts(self):
    root_id = generate_ids.gen_root_id()
    base = TransformGroup('BASE', root_id)
    this = TransformGroup('THIS', root_id)
    other = TransformGroup('OTHER', root_id)
    base_a, this_a, other_a = (t.tt.new_directory('a', t.root, b'a') for t in [base, this, other])
    base_b, this_b, other_b = (t.tt.new_directory('b', t.root, b'b') for t in [base, this, other])
    base.tt.new_file('g', base_a, [b'g'], b'g')
    other.tt.new_file('g1', other_b, [b'g1'], b'g')
    base.tt.new_file('h', base_a, [b'h'], b'h')
    this.tt.new_file('h1', this_b, [b'h1'], b'h')
    base.tt.new_file('i', base.root, [b'i'], b'i')
    other.tt.new_directory('i1', this_b, b'i')
    for tg in [this, base, other]:
        tg.tt.apply()
    Merge3Merger(this.wt, this.wt, base.wt, other.wt)
    self.assertEqual(this.wt.id2path(b'g'), pathjoin('b/g1.OTHER'))
    self.assertIs(os.path.lexists(this.wt.abspath('b/g1.BASE')), True)
    self.assertIs(os.path.lexists(this.wt.abspath('b/g1.THIS')), False)
    self.assertEqual(this.wt.id2path(b'h'), pathjoin('b/h1.THIS'))
    self.assertIs(os.path.lexists(this.wt.abspath('b/h1.BASE')), True)
    self.assertIs(os.path.lexists(this.wt.abspath('b/h1.OTHER')), False)
    self.assertEqual(this.wt.id2path(b'i'), pathjoin('b/i1.OTHER'))