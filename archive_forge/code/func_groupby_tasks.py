from __future__ import annotations
import io
import itertools
import math
import operator
import uuid
import warnings
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from functools import partial, reduce, wraps
from random import Random
from urllib.request import urlopen
import tlz as toolz
from fsspec.core import open_files
from tlz import (
from dask import config
from dask.bag import chunk
from dask.bag.avro import to_avro
from dask.base import (
from dask.blockwise import blockwise
from dask.context import globalmethod
from dask.core import flatten, get_dependencies, istask, quote, reverse_dict
from dask.delayed import Delayed, unpack_collections
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import cull, fuse, inline
from dask.sizeof import sizeof
from dask.typing import Graph, NestedKeys, no_default
from dask.utils import (
def groupby_tasks(b, grouper, hash=lambda x: int(tokenize(x), 16), max_branch=32):
    max_branch = max_branch or 32
    n = b.npartitions
    stages = int(math.ceil(math.log(n) / math.log(max_branch))) or 1
    if stages > 1:
        k = int(math.ceil(n ** (1 / stages)))
    else:
        k = n
    groups = []
    splits = []
    joins = []
    inputs = [tuple((digit(i, j, k) for j in range(stages))) for i in range(k ** stages)]
    b2 = b.map(partial(chunk.groupby_tasks_group_hash, hash=hash, grouper=grouper))
    token = tokenize(b, grouper, hash, max_branch)
    shuffle_join_name = 'shuffle-join-' + token
    shuffle_group_name = 'shuffle-group-' + token
    shuffle_split_name = 'shuffle-split-' + token
    start = {}
    for idx, inp in enumerate(inputs):
        group = {}
        split = {}
        if idx < b.npartitions:
            start[shuffle_join_name, 0, inp] = (b2.name, idx)
        else:
            start[shuffle_join_name, 0, inp] = []
        for stage in range(1, stages + 1):
            _key_tuple = (shuffle_group_name, stage, inp)
            group[_key_tuple] = (groupby, (make_group, k, stage - 1), (shuffle_join_name, stage - 1, inp))
            for i in range(k):
                split[shuffle_split_name, stage, i, inp] = (dict.get, _key_tuple, i, {})
        groups.append(group)
        splits.append(split)
    for stage in range(1, stages + 1):
        join = {(shuffle_join_name, stage, inp): (list, (toolz.concat, [(shuffle_split_name, stage, inp[stage - 1], insert(inp, stage - 1, j)) for j in range(k)])) for inp in inputs}
        joins.append(join)
    name = 'shuffle-' + token
    end = {(name, i): (list, (dict.items, (groupby, grouper, (pluck, 1, j)))) for i, j in enumerate(join)}
    groups.extend(splits)
    groups.extend(joins)
    dsk = merge(start, end, *groups)
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[b2])
    return type(b)(graph, name, len(inputs))