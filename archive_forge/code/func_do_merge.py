import os
from .. import errors
from .. import merge as _mod_merge
from ..bzr.conflicts import TextConflict
from . import TestCaseWithTransport, multiply_tests
from .test_merge_core import MergeBuilder
def do_merge(self, target_tree, source_tree, **kwargs):
    merger = _mod_merge.Merger.from_revision_ids(target_tree, source_tree.last_revision(), other_branch=source_tree.branch)
    merger.merge_type = self.merge_type
    for name, value in kwargs.items():
        setattr(merger, name, value)
    merger.do_merge()