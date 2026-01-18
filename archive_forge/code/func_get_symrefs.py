import os
import warnings
from contextlib import suppress
from typing import Any, Dict, Optional, Set
from .errors import PackedRefsException, RefFormatError
from .file import GitFile, ensure_dir_exists
from .objects import ZERO_SHA, ObjectID, Tag, git_line, valid_hexsha
from .pack import ObjectContainer
def get_symrefs(self):
    """Get a dict with all symrefs in this container.

        Returns: Dictionary mapping source ref to target ref
        """
    ret = {}
    for src in self.allkeys():
        try:
            dst = parse_symref_value(self.read_ref(src))
        except ValueError:
            pass
        else:
            ret[src] = dst
    return ret