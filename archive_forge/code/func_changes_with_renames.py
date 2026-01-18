import stat
from collections import defaultdict, namedtuple
from io import BytesIO
from itertools import chain
from typing import Dict, List, Optional
from .objects import S_ISGITLINK, Tree, TreeEntry
def changes_with_renames(self, tree1_id, tree2_id, want_unchanged=False, include_trees=False):
    """Iterate TreeChanges between two tree SHAs, with rename detection."""
    self._reset()
    self._want_unchanged = want_unchanged
    self._include_trees = include_trees
    self._collect_changes(tree1_id, tree2_id)
    self._find_exact_renames()
    self._find_content_rename_candidates()
    self._choose_content_renames()
    self._join_modifies()
    self._prune_unchanged()
    return self._sorted_changes()