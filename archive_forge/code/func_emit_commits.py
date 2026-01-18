import re
import sys
import time
from email.utils import parseaddr
import breezy.branch
import breezy.revision
from ... import (builtins, errors, lazy_import, lru_cache, osutils, progress,
from ... import transport as _mod_transport
from . import helpers, marks_file
from fastimport import commands
def emit_commits(self, interesting):
    if self.baseline:
        revobj = self.branch.repository.get_revision(interesting.pop(0))
        self.emit_baseline(revobj, self.ref)
    for i in range(0, len(interesting), REVISIONS_CHUNK_SIZE):
        chunk = interesting[i:i + REVISIONS_CHUNK_SIZE]
        history = dict(self.branch.repository.iter_revisions(chunk))
        trees_needed = set()
        trees = {}
        for revid in chunk:
            trees_needed.update(self.preprocess_commit(revid, history[revid], self.ref))
        for tree in self._get_revision_trees(trees_needed):
            trees[tree.get_revision_id()] = tree
        for revid in chunk:
            revobj = history[revid]
            if len(revobj.parent_ids) == 0:
                parent = breezy.revision.NULL_REVISION
            else:
                parent = revobj.parent_ids[0]
            self.emit_commit(revobj, self.ref, trees[parent], trees[revid])