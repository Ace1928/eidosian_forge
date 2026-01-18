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
def test_criss_cross_flip_flop(self):
    self.add_rev(b'root', b'XX', [], b'qrs')
    self.add_rev(b'root', b'A', [b'XX'], b'abcdef')
    self.add_rev(b'root', b'B', [b'A'], b'abcdgef')
    self.add_rev(b'root', b'C', [b'A'], b'abcdhef')
    self.add_rev(b'root', b'D', [b'B', b'C'], b'abcdghef')
    self.add_rev(b'root', b'E', [b'C', b'B'], b'abcdhgef')
    plan = list(self.plan_merge_vf.plan_merge(b'D', b'E'))
    self.assertEqual([('unchanged', b'a\n'), ('unchanged', b'b\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('new-b', b'h\n'), ('unchanged', b'g\n'), ('killed-b', b'h\n'), ('unchanged', b'e\n'), ('unchanged', b'f\n')], plan)
    pwm = versionedfile.PlanWeaveMerge(plan)
    self.assertEqualDiff(b'a\nb\nc\nd\ng\nh\ne\nf\n', b''.join(pwm.base_from_plan()))
    plan = list(self.plan_merge_vf.plan_merge(b'E', b'D'))
    self.assertEqual([('unchanged', b'a\n'), ('unchanged', b'b\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('new-b', b'g\n'), ('unchanged', b'h\n'), ('killed-b', b'g\n'), ('unchanged', b'e\n'), ('unchanged', b'f\n')], plan)
    pwm = versionedfile.PlanWeaveMerge(plan)
    self.assertEqualDiff(b'a\nb\nc\nd\nh\ng\ne\nf\n', b''.join(pwm.base_from_plan()))
    plan = list(self.plan_merge_vf.plan_lca_merge(b'D', b'E'))
    self.assertEqual([('unchanged', b'a\n'), ('unchanged', b'b\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('conflicted-b', b'h\n'), ('unchanged', b'g\n'), ('conflicted-a', b'h\n'), ('unchanged', b'e\n'), ('unchanged', b'f\n')], plan)
    pwm = versionedfile.PlanWeaveMerge(plan)
    self.assertEqualDiff(b'a\nb\nc\nd\ng\ne\nf\n', b''.join(pwm.base_from_plan()))
    plan = list(self.plan_merge_vf.plan_lca_merge(b'E', b'D'))
    self.assertEqual([('unchanged', b'a\n'), ('unchanged', b'b\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('conflicted-b', b'g\n'), ('unchanged', b'h\n'), ('conflicted-a', b'g\n'), ('unchanged', b'e\n'), ('unchanged', b'f\n')], plan)
    pwm = versionedfile.PlanWeaveMerge(plan)
    self.assertEqualDiff(b'a\nb\nc\nd\nh\ne\nf\n', b''.join(pwm.base_from_plan()))