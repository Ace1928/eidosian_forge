from typing import FrozenSet, Optional, Set, Tuple
import gevent
from gevent import pool
from .object_store import (
from .objects import Commit, ObjectID, Tag
def collect_tree_sha(sha):
    self.sha_done.add(sha)
    cmt = object_store[sha]
    _collect_filetree_revs(object_store, cmt.tree, self.sha_done)