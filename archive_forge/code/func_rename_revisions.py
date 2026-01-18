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
def rename_revisions(self, revid_map):
    self._tag_dict = {name: revid_map.get(revid, revid) for name, revid in self._tag_dict.items()}