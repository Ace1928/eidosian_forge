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
def test_all_wt(self):
    """Check behavior if all trees are Working Trees."""
    builder = self.get_builder()
    builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'base content\n')), ('add', ('foo', b'foo-id', 'file', b'base content\n'))], revision_id=b'A-id')
    builder.build_snapshot([b'A-id'], [('modify', ('foo', b'B content\n'))], revision_id=b'B-id')
    builder.build_snapshot([b'A-id'], [('rename', ('a', 'b'))], revision_id=b'C-id')
    builder.build_snapshot([b'C-id', b'B-id'], [('rename', ('b', 'c')), ('modify', ('foo', b'E content\n'))], revision_id=b'E-id')
    builder.build_snapshot([b'B-id', b'C-id'], [('rename', ('a', 'b'))], revision_id=b'D-id')
    wt_this = self.get_wt_from_builder(builder)
    wt_base = wt_this.controldir.sprout('base', b'A-id').open_workingtree()
    wt_base.lock_read()
    self.addCleanup(wt_base.unlock)
    wt_lca1 = wt_this.controldir.sprout('b-tree', b'B-id').open_workingtree()
    wt_lca1.lock_read()
    self.addCleanup(wt_lca1.unlock)
    wt_lca2 = wt_this.controldir.sprout('c-tree', b'C-id').open_workingtree()
    wt_lca2.lock_read()
    self.addCleanup(wt_lca2.unlock)
    wt_other = wt_this.controldir.sprout('other', b'E-id').open_workingtree()
    wt_other.lock_read()
    self.addCleanup(wt_other.unlock)
    merge_obj = _mod_merge.Merge3Merger(wt_this, wt_this, wt_base, wt_other, lca_trees=[wt_lca1, wt_lca2], do_merge=False)
    entries = list(merge_obj._entries_lca())
    root_id = b'a-root-id'
    self.assertEqual([(b'a-id', False, (('a', ['a', 'b']), 'c', 'b'), ((root_id, [root_id, root_id]), root_id, root_id), (('a', ['a', 'b']), 'c', 'b'), ((False, [False, False]), False, False), False), (b'foo-id', True, (('foo', ['foo', 'foo']), 'foo', 'foo'), ((root_id, [root_id, root_id]), root_id, root_id), (('foo', ['foo', 'foo']), 'foo', 'foo'), ((False, [False, False]), False, False), False)], entries)