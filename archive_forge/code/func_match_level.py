import difflib
import inspect
import pickle
import traceback
from collections import defaultdict
from contextlib import contextmanager
import numpy as np
import param
from .accessors import Opts  # noqa (clean up in 2.0)
from .pprint import InfoPrinter
from .tree import AttrTree
from .util import group_sanitizer, label_sanitizer, sanitize_identifier
def match_level(self, overlay):
    """
        Given an overlay, return the match level and applicable slice
        of the overall overlay. The level an integer if there is a
        match or None if there is no match.

        The level integer is the number of matching components. Higher
        values indicate a stronger match.
        """
    slice_width = len(self._pattern_spec)
    if slice_width > len(overlay):
        return None
    best_lvl, match_slice = (0, None)
    for i in range(len(overlay) - slice_width + 1):
        overlay_slice = overlay.values()[i:i + slice_width]
        lvl = self._slice_match_level(overlay_slice)
        if lvl is None:
            continue
        if lvl > best_lvl:
            best_lvl = lvl
            match_slice = (i, i + slice_width)
    return (best_lvl, match_slice) if best_lvl != 0 else None