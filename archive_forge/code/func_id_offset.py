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
def id_offset(cls):
    """
        Compute an appropriate offset for future id values given the set
        of ids currently defined across backends.
        """
    max_ids = []
    for backend in Store.renderers.keys():
        store_ids = list(Store.custom_options(backend=backend).keys())
        max_id = max(store_ids) + 1 if len(store_ids) > 0 else 0
        max_ids.append(max_id)
    return max(max_ids) if len(max_ids) else 0