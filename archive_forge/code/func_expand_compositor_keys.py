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
def expand_compositor_keys(cls, spec):
    """
        Expands compositor definition keys into {type}.{group}
        keys. For instance a compositor operation returning a group
        string 'Image' of element type RGB expands to 'RGB.Image'.
        """
    expanded_spec = {}
    applied_keys = []
    compositor_defs = {el.group: el.output_type.__name__ for el in Compositor.definitions}
    for key, val in spec.items():
        if key not in compositor_defs:
            expanded_spec[key] = val
        else:
            applied_keys = ['Overlay']
            type_name = compositor_defs[key]
            expanded_spec[str(type_name + '.' + key)] = val
    return (expanded_spec, applied_keys)