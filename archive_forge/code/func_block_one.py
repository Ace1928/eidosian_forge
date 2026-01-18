from __future__ import annotations
import uuid
from collections.abc import Callable, Hashable
from typing import Literal, TypeVar
from dask.base import (
from dask.blockwise import blockwise
from dask.core import flatten
from dask.delayed import Delayed, delayed
from dask.highlevelgraph import HighLevelGraph, Layer, MaterializedLayer
from dask.typing import Graph, Key
def block_one(coll):
    tok = tokenize(coll, blocker)
    dsks = []
    rename = {}
    for prev_name in get_collection_names(coll):
        new_name = 'wait_on-' + tokenize(prev_name, tok)
        rename[prev_name] = new_name
        layer = _build_map_layer(chunks.bind, prev_name, new_name, coll, dependencies=(blocker,))
        dsks.append(HighLevelGraph.from_collections(new_name, layer, dependencies=(coll, blocker)))
    dsk = HighLevelGraph.merge(*dsks)
    rebuild, args = coll.__dask_postpersist__()
    return rebuild(dsk, *args, rename=rename)