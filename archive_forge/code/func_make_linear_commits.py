from itertools import permutations
from unittest import expectedFailure
from dulwich.tests import TestCase
from ..diff_tree import CHANGE_MODIFY, CHANGE_RENAME, RenameDetector, TreeChange
from ..errors import MissingCommitError
from ..object_store import MemoryObjectStore
from ..objects import Blob, Commit
from ..walk import ORDER_TOPO, WalkEntry, Walker, _topo_reorder
from .utils import F, build_commit_graph, make_object, make_tag
def make_linear_commits(self, num_commits, **kwargs):
    commit_spec = []
    for i in range(1, num_commits + 1):
        c = [i]
        if i > 1:
            c.append(i - 1)
        commit_spec.append(c)
    return self.make_commits(commit_spec, **kwargs)