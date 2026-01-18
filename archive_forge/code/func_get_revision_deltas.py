from typing import List, Type, TYPE_CHECKING, Optional, Iterable
from .lazy_import import lazy_import
import time
from breezy import (
from breezy.i18n import gettext
from . import controldir, debug, errors, graph, registry, revision as _mod_revision, ui
from .decorators import only_raises
from .inter import InterObject
from .lock import LogicalLockResult, _RelockDebugMixin
from .revisiontree import RevisionTree
from .trace import (log_exception_quietly, mutter, mutter_callsite, note,
def get_revision_deltas(self, revisions, specific_files=None):
    """Produce a generator of revision deltas.

        Note that the input is a sequence of REVISIONS, not revision ids.
        Trees will be held in memory until the generator exits.
        Each delta is relative to the revision's lefthand predecessor.

        specific_files should exist in the first revision.

        Args:
          specific_files: if not None, the result is filtered
          so that only those files, their parents and their
          children are included.
        """
    from .tree import InterTree
    required_trees = set()
    for revision in revisions:
        required_trees.add(revision.revision_id)
        required_trees.update(revision.parent_ids[:1])
    trees = {t.get_revision_id(): t for t in self.revision_trees(required_trees)}
    for revision in revisions:
        if not revision.parent_ids:
            old_tree = self.revision_tree(_mod_revision.NULL_REVISION)
        else:
            old_tree = trees[revision.parent_ids[0]]
        intertree = InterTree.get(old_tree, trees[revision.revision_id])
        yield intertree.compare(specific_files=specific_files)
        if specific_files is not None:
            specific_files = [p for p in intertree.find_source_paths(specific_files).values() if p is not None]