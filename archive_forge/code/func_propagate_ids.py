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
def propagate_ids(cls, obj, match_id, new_id, applied_keys, backend=None):
    """
        Recursively propagate an id through an object for components
        matching the applied_keys. This method can only be called if
        there is a tree with a matching id in Store.custom_options
        """
    applied = []

    def propagate(o):
        if o.id == match_id or o.__class__.__name__ == 'DynamicMap':
            o.id = new_id
            applied.append(o)
    obj.traverse(propagate, specs=set(applied_keys) | {'DynamicMap'})
    if new_id not in Store.custom_options(backend=backend):
        raise AssertionError('New option id %d does not match any option trees in Store.custom_options.' % new_id)
    return applied