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
def setup_simple_branch(self, relpath, shape=None, root_id=None):
    """One commit, containing tree specified by optional shape.

        Default is empty tree (just root entry).
        """
    if root_id is None:
        root_id = b'%s-root-id' % (relpath.encode('ascii'),)
    wt = self.make_branch_and_tree(relpath)
    wt.set_root_id(root_id)
    if shape is not None:
        adjusted_shape = [relpath + '/' + elem for elem in shape]
        self.build_tree(adjusted_shape)
        ids = [b'%s-%s-id' % (relpath.encode('utf-8'), basename(elem.rstrip('/')).encode('ascii')) for elem in shape]
        wt.add(shape, ids=ids)
    rev_id = b'r1-%s' % (relpath.encode('utf-8'),)
    wt.commit('Initial commit of {}'.format(relpath), rev_id=rev_id)
    self.assertEqual(root_id, wt.path2id(''))
    return wt