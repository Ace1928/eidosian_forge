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
def test_deserialize_creation(self):
    tt = self.get_preview()
    tt.deserialize(iter(self.creation_records()))
    self.assertEqual(3, tt._id_number)
    self.assertEqual({'new-1': 'fooሴ', 'new-2': 'qux'}, tt._new_name)
    self.assertEqual({'new-1': b'baz', 'new-2': b'quxx'}, tt._new_id)
    self.assertEqual({'new-1': tt.root, 'new-2': tt.root}, tt._new_parent)
    self.assertEqual({b'baz': 'new-1', b'quxx': 'new-2'}, tt._r_new_id)
    self.assertEqual({'new-1': True}, tt._new_executability)
    self.assertEqual({'new-1': 'file', 'new-2': 'directory'}, tt._new_contents)
    with open(tt._limbo_name('new-1'), 'rb') as foo_limbo:
        foo_content = foo_limbo.read()
    self.assertEqual(b'bar', foo_content)