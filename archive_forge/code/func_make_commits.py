from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def make_commits(self, commit_spec, **kwargs):
    times = kwargs.pop('times', [])
    attrs = kwargs.pop('attrs', {})
    for i, t in enumerate(times):
        attrs.setdefault(i + 1, {})['commit_time'] = t
    return build_commit_graph(self.store, commit_spec, attrs=attrs, **kwargs)