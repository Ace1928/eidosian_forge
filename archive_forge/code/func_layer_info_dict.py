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
def layer_info_dict(self):
    info = {'layer_type': type(self).__name__, 'is_materialized': self.is_materialized(), 'number of outputs': f'{len(self.get_output_keys())}'}
    if self.annotations is not None:
        for key, val in self.annotations.items():
            info[key] = html.escape(str(val))
    if self.collection_annotations is not None:
        for key, val in self.collection_annotations.items():
            if key != 'chunks':
                info[key] = html.escape(str(val))
    return info