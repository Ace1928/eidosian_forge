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
def restore_ids(cls, obj, ids):
    """
        Given an list of ids as captured with capture_ids, restore the
        ids. Note the structure of an object must not change between
        the calls to capture_ids and restore_ids.
        """
    ids = iter(ids)
    obj.traverse(lambda o: setattr(o, 'id', next(ids)))