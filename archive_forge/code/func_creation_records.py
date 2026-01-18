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
def creation_records(self):
    attribs = self.default_attribs()
    attribs[b'_id_number'] = 3
    attribs[b'_new_name'] = {b'new-1': 'fooሴ'.encode(), b'new-2': b'qux'}
    attribs[b'_new_id'] = {b'new-1': b'baz', b'new-2': b'quxx'}
    attribs[b'_new_parent'] = {b'new-1': b'new-0', b'new-2': b'new-0'}
    attribs[b'_new_executability'] = {b'new-1': 1}
    contents = [(b'new-1', b'file', b'i 1\nbar\n'), (b'new-2', b'directory', b'')]
    return self.make_records(attribs, contents)