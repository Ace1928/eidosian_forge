import contextlib
import itertools
import re
import sys
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple
from . import branch as _mod_branch
from . import errors
from .inter import InterObject
from .registry import Registry
from .revision import RevisionID
def sort_time(branch, tags):
    """Sort tags by time inline.

    :param branch: Branch
    :param tags: List of tuples with tag name and revision id.
    """
    timestamps = {}
    for tag, revid in tags:
        try:
            revobj = branch.repository.get_revision(revid)
        except errors.NoSuchRevision:
            timestamp = sys.maxsize
        else:
            timestamp = revobj.timestamp
        timestamps[revid] = timestamp
    tags.sort(key=lambda x: timestamps[x[1]])