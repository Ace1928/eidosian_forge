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
def compute_layer_dependencies(layers):
    """Returns the dependencies between layers"""

    def _find_layer_containing_key(key):
        for k, v in layers.items():
            if key in v:
                return k
        raise RuntimeError(f'{repr(key)} not found')
    all_keys = {key for layer in layers.values() for key in layer}
    ret = {k: set() for k in layers}
    for k, v in layers.items():
        for key in keys_in_tasks(all_keys - v.keys(), v.values()):
            ret[k].add(_find_layer_containing_key(key))
    return ret