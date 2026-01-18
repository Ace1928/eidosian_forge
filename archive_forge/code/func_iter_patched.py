import os
import re
from typing import Iterator, List, Optional
from .errors import BzrError
def iter_patched(orig_lines, patch_lines):
    """Iterate through a series of lines with a patch applied.
    This handles a single file, and does exact, not fuzzy patching.
    """
    patch_lines = iter_lines_handle_nl(iter(patch_lines))
    get_patch_names(patch_lines)
    return iter_patched_from_hunks(orig_lines, iter_hunks(patch_lines))