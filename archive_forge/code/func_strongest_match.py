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
@classmethod
def strongest_match(cls, overlay, mode, backend=None):
    """
        Returns the single strongest matching compositor operation
        given an overlay. If no matches are found, None is returned.

        The best match is defined as the compositor operation with the
        highest match value as returned by the match_level method.
        """
    match_strength = [(op.match_level(overlay), op) for op in cls.definitions if op.mode == mode and (not op.backends or backend in op.backends)]
    matches = [(match[0], op, match[1]) for match, op in match_strength if match is not None]
    if matches == []:
        return None
    return sorted(matches)[0]