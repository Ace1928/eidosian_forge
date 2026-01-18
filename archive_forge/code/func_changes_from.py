from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
def changes_from(self, other, want_unchanged=False, specific_files=None, extra_trees=None, require_versioned=False, include_root=False, want_unversioned=False):
    """Return a TreeDelta of the changes from other to this tree.

        Args:
          other: A tree to compare with.
          specific_files: An optional list of file paths to restrict the
            comparison to. When mapping filenames to ids, all matches in all
            trees (including optional extra_trees) are used, and all children of
            matched directories are included.
          want_unchanged: An optional boolean requesting the inclusion of
            unchanged entries in the result.
          extra_trees: An optional list of additional trees to use when
            mapping the contents of specific_files (paths) to their identities.
          require_versioned: An optional boolean (defaults to False). When
            supplied and True all the 'specific_files' must be versioned, or
            a PathsNotVersionedError will be thrown.
          want_unversioned: Scan for unversioned paths.

        The comparison will be performed by an InterTree object looked up on
        self and other.
        """
    return InterTree.get(other, self).compare(want_unchanged=want_unchanged, specific_files=specific_files, extra_trees=extra_trees, require_versioned=require_versioned, include_root=include_root, want_unversioned=want_unversioned)