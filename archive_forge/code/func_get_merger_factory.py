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
def get_merger_factory(self):
    calls = self.calls

    class FooMerger(_mod_merge.ConfigurableFileMerger):
        name_prefix = 'foo'
        default_files = ['bar']

        def merge_text(self, params):
            calls.append('merge_text')
            return ('not_applicable', None)

    def factory(merger):
        result = FooMerger(merger)
        self.assertEqual(None, result.affected_files)
        self.merger = result
        return result
    return factory