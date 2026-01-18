import os
import re
from collections import deque
from typing import TYPE_CHECKING, Optional, Type
from .. import branch as _mod_branch
from .. import controldir, debug, errors, lazy_import, osutils, revision, trace
from .. import transport as _mod_transport
from ..controldir import ControlDir
from ..mutabletree import MutableTree
from ..repository import Repository
from ..revisiontree import RevisionTree
from breezy import (
from breezy.bzr import (
from ..tree import (FileTimestampUnavailable, InterTree, MissingNestedTree,
def discard_new(self):
    return self.__class__(self.file_id, (self.path[0], None), self.changed_content, (self.versioned[0], None), (self.parent_id[0], None), (self.name[0], None), (self.kind[0], None), (self.executable[0], None), copied=False)