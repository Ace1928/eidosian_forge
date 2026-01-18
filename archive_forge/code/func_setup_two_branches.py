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
def setup_two_branches(self, custom_root_ids=True):
    """Setup 2 branches, one will be a library, the other a project."""
    if custom_root_ids:
        root_id = None
    else:
        root_id = inventory.ROOT_ID
    project_wt = self.setup_simple_branch('project', ['README', 'dir/', 'dir/file.c'], root_id)
    lib_wt = self.setup_simple_branch('lib1', ['README', 'Makefile', 'foo.c'], root_id)
    return (project_wt, lib_wt)