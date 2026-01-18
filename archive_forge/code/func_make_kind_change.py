import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
def make_kind_change(self):
    factory = self.get_merger_factory()
    self._install_hook(factory)
    builder = self.make_builder()
    builder.add_file(builder.root(), 'bar', b'text1', True, this=False, file_id=b'bar-id')
    builder.add_dir(builder.root(), 'bar-id', base=False, other=False, file_id=b'bar-dir')
    return builder