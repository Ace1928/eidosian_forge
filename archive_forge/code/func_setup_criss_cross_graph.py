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
def setup_criss_cross_graph(self):
    """Create a 5-node graph with a criss-cross.

        :return: A BranchBuilder
        """
    builder = self.setup_simple_graph()
    builder.build_snapshot([b'C-id', b'B-id'], [], revision_id=b'E-id')
    builder.build_snapshot([b'B-id', b'C-id'], [], revision_id=b'D-id')
    return builder