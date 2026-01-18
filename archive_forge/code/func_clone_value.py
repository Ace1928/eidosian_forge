from __future__ import annotations
import abc
import copy
import html
from collections.abc import (
from typing import Any
import tlz as toolz
import dask
from dask import config
from dask.base import clone_key, flatten, is_dask_collection, normalize_token
from dask.core import keys_in_tasks, reverse_dict
from dask.typing import DaskCollection, Graph, Key
from dask.utils import ensure_dict, import_required, key_split
from dask.widgets import get_template
def clone_value(o):
    """Variant of distributed.utils_comm.subs_multiple, which allows injecting
            bind_to
            """
    nonlocal is_leaf
    typ = type(o)
    if typ is tuple and o and callable(o[0]):
        return (o[0],) + tuple((clone_value(i) for i in o[1:]))
    elif typ is list:
        return [clone_value(i) for i in o]
    elif typ is dict:
        return {k: clone_value(v) for k, v in o.items()}
    else:
        try:
            if o not in keys:
                return o
        except TypeError:
            return o
        is_leaf = False
        return clone_key(o, seed)