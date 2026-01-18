import collections
import os
import stat
import struct
import sys
from typing import (
from dulwich.file import GitFile
from dulwich.objects import (
from dulwich.pack import (
def iter_fresh_blobs(index, root_path):
    """Iterate over versions of blobs on disk referenced by index.

    Don't use this function; it removes missing entries from index.

    Args:
      index: Index file
      root_path: Root path to access from
      include_deleted: Include deleted entries with sha and
        mode set to None
    Returns: Iterator over path, sha, mode
    """
    import warnings
    warnings.warn(PendingDeprecationWarning, 'Use iter_fresh_objects instead.')
    for entry in iter_fresh_objects(index, root_path, include_deleted=True):
        if entry[1] is None:
            del index[entry[0]]
        else:
            yield entry