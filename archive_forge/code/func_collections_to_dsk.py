from __future__ import annotations
import dataclasses
import datetime
import decimal
import hashlib
import inspect
import pathlib
import pickle
import types
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Iterator, Mapping
from concurrent.futures import Executor
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from functools import partial
from numbers import Integral, Number
from operator import getitem
from typing import Any, Literal, TypeVar
import cloudpickle
from tlz import curry, groupby, identity, merge
from tlz.functoolz import Compose
from dask import config, local
from dask._compatibility import EMSCRIPTEN
from dask.core import flatten
from dask.core import get as simple_get
from dask.core import literal, quote
from dask.hashing import hash_buffer_hex
from dask.system import CPU_COUNT
from dask.typing import Key, SchedulerGetCallable
from dask.utils import (
def collections_to_dsk(collections, optimize_graph=True, optimizations=(), **kwargs):
    """
    Convert many collections into a single dask graph, after optimization
    """
    from dask.highlevelgraph import HighLevelGraph
    optimizations = tuple(optimizations) + tuple(config.get('optimizations', ()))
    if optimize_graph:
        groups = groupby(optimization_function, collections)
        graphs = []
        for opt, val in groups.items():
            dsk, keys = _extract_graph_and_keys(val)
            dsk = opt(dsk, keys, **kwargs)
            for opt_inner in optimizations:
                dsk = opt_inner(dsk, keys, **kwargs)
            graphs.append(dsk)
        if any((isinstance(graph, HighLevelGraph) for graph in graphs)):
            dsk = HighLevelGraph.merge(*graphs)
        else:
            dsk = merge(*map(ensure_dict, graphs))
    else:
        dsk, _ = _extract_graph_and_keys(collections)
    return dsk