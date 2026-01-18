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
def sort_natural(branch, tags):
    """Sort tags, with numeric substrings as numbers.

    :param branch: Branch
    :param tags: List of tuples with tag name and revision id.
    """

    def natural_sort_key(tag):
        return [f(s) for f, s in zip(itertools.cycle((str.lower, int)), re.split('([0-9]+)', tag[0]))]
    tags.sort(key=natural_sort_key)