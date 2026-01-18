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
def plan_file_lca_merge(self, path, other, base=None):
    """Generate a merge plan based lca-newness.

        If the file contains uncommitted changes in this tree, they will be
        attributed to the 'current:' pseudo-revision.  If the file contains
        uncommitted changes in the other tree, they will be assigned to the
        'other:' pseudo-revision.
        """
    data = self._get_plan_merge_data(path, other, base)
    vf, last_revision_a, last_revision_b, last_revision_base = data
    return vf.plan_lca_merge(last_revision_a, last_revision_b, last_revision_base)