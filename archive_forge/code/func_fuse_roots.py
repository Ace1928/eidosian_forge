from __future__ import annotations
import itertools
import os
from collections.abc import Hashable, Iterable, Mapping, Sequence
from itertools import product
from math import prod
from typing import Any
import tlz as toolz
import dask
from dask.base import clone_key, get_name_from_key, tokenize
from dask.core import flatten, ishashable, keys_in_tasks, reverse_dict
from dask.highlevelgraph import HighLevelGraph, Layer
from dask.optimization import SubgraphCallable, fuse
from dask.typing import Graph, Key
from dask.utils import (
def fuse_roots(graph: HighLevelGraph, keys: list):
    """
    Fuse nearby layers if they don't have dependencies

    Often Blockwise sections of the graph fill out all of the computation
    except for the initial data access or data loading layers::

      Large Blockwise Layer
        |       |       |
        X       Y       Z

    This can be troublesome because X, Y, and Z tasks may be executed on
    different machines, and then require communication to move around.

    This optimization identifies this situation, lowers all of the graphs to
    concrete dicts, and then calls ``fuse`` on them, with a width equal to the
    number of layers like X, Y, and Z.

    This is currently used within array and dataframe optimizations.

    Parameters
    ----------
    graph : HighLevelGraph
        The full graph of the computation
    keys : list
        The output keys of the computation, to be passed on to fuse

    See Also
    --------
    Blockwise
    fuse
    """
    layers = ensure_dict(graph.layers, copy=True)
    dependencies = ensure_dict(graph.dependencies, copy=True)
    dependents = reverse_dict(dependencies)
    for name, layer in graph.layers.items():
        deps = graph.dependencies[name]
        if isinstance(layer, Blockwise) and len(deps) > 1 and (not any((dependencies[dep] for dep in deps))) and all((len(dependents[dep]) == 1 for dep in deps)) and all((layer.annotations == graph.layers[dep].annotations for dep in deps)):
            new = toolz.merge(layer, *[layers[dep] for dep in deps])
            new, _ = fuse(new, keys, ave_width=len(deps))
            for dep in deps:
                del layers[dep]
                del dependencies[dep]
            layers[name] = new
            dependencies[name] = set()
    return HighLevelGraph(layers, dependencies)