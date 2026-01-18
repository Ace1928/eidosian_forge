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
def cleanup_custom_options(id, weakref=None):
    """
    Cleans up unused custom trees if all objects referencing the
    custom id have been garbage collected or tree is otherwise
    unreferenced.
    """
    try:
        if Store._options_context:
            return
        weakrefs = Store._weakrefs.get(id, [])
        if weakref in weakrefs:
            weakrefs.remove(weakref)
        refs = []
        for wr in list(weakrefs):
            r = wr()
            if r is None or r.id != id:
                weakrefs.remove(wr)
            else:
                refs.append(r)
        if not refs:
            for bk in Store.loaded_backends():
                if id in Store._custom_options[bk]:
                    Store._custom_options[bk].pop(id)
        if not weakrefs:
            Store._weakrefs.pop(id, None)
    except Exception as e:
        raise Exception(f"Cleanup of custom options tree with id '{id}' failed with the following exception: {e}, an unreferenced orphan tree may persist in memory.") from e