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
def cull_layers(self, layers: Iterable[str]) -> HighLevelGraph:
    """Return a new HighLevelGraph with only the given layers and their
        dependencies. Internally, layers are not modified.

        This is a variant of :meth:`HighLevelGraph.cull` which is much faster and does
        not risk creating a collision between two layers with the same name and
        different content when two culled graphs are merged later on.

        Returns
        -------
        hlg: HighLevelGraph
            Culled high level graph
        """
    to_visit = set(layers)
    ret_layers = {}
    ret_dependencies = {}
    while to_visit:
        k = to_visit.pop()
        ret_layers[k] = self.layers[k]
        ret_dependencies[k] = self.dependencies[k]
        to_visit |= ret_dependencies[k] - ret_dependencies.keys()
    return HighLevelGraph(ret_layers, ret_dependencies)