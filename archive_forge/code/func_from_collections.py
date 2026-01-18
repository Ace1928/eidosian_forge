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
@classmethod
def from_collections(cls, name: str, layer: Graph, dependencies: Sequence[DaskCollection]=()) -> HighLevelGraph:
    """Construct a HighLevelGraph from a new layer and a set of collections

        This constructs a HighLevelGraph in the common case where we have a single
        new layer and a set of old collections on which we want to depend.

        This pulls out the ``__dask_layers__()`` method of the collections if
        they exist, and adds them to the dependencies for this new layer.  It
        also merges all of the layers from all of the dependent collections
        together into the new layers for this graph.

        Parameters
        ----------
        name : str
            The name of the new layer
        layer : Mapping
            The graph layer itself
        dependencies : List of Dask collections
            A list of other dask collections (like arrays or dataframes) that
            have graphs themselves

        Examples
        --------

        In typical usage we make a new task layer, and then pass that layer
        along with all dependent collections to this method.

        >>> def add(self, other):
        ...     name = 'add-' + tokenize(self, other)
        ...     layer = {(name, i): (add, input_key, other)
        ...              for i, input_key in enumerate(self.__dask_keys__())}
        ...     graph = HighLevelGraph.from_collections(name, layer, dependencies=[self])
        ...     return new_collection(name, graph)
        """
    if len(dependencies) == 1:
        return cls._from_collection(name, layer, dependencies[0])
    layers = {name: layer}
    name_dep: set[str] = set()
    deps: dict[str, set[str]] = {name: name_dep}
    for collection in toolz.unique(dependencies, key=id):
        if is_dask_collection(collection):
            graph = collection.__dask_graph__()
            if isinstance(graph, HighLevelGraph):
                layers.update(graph.layers)
                deps.update(graph.dependencies)
                name_dep |= set(collection.__dask_layers__())
            else:
                key = _get_some_layer_name(collection)
                layers[key] = graph
                name_dep.add(key)
                deps[key] = set()
        else:
            raise TypeError(type(collection))
    return cls(layers, deps)