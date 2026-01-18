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
def make_text_conflict(self, file_name='bar'):
    factory = self.get_merger_factory()
    self._install_hook(factory)
    builder = self.make_builder()
    trans_ids = builder.add_file(builder.root(), file_name, b'text1', True, file_id=b'bar-id')
    builder.change_contents(trans_ids, other=b'text4', this=b'text3')
    return builder