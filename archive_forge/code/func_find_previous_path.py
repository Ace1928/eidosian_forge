from typing import (TYPE_CHECKING, Dict, Iterator, List, Optional, Type, Union,
from . import errors, lock, osutils
from . import revision as _mod_revision
from . import trace
from .inter import InterObject
def find_previous_path(from_tree, to_tree, path, recurse='none'):
    """Find previous tree path.

    Args:
      from_tree: From tree
      to_tree: To tree
      path: Path to search for (exists in from_tree)
    Returns: path in to_tree, or None if there is no equivalent path.
    Raises:
      NoSuchFile: If the path doesn't exist in from_tree
    """
    return InterTree.get(to_tree, from_tree).find_source_path(path, recurse=recurse)