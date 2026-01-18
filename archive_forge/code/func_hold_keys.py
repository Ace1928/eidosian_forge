from __future__ import annotations
from collections.abc import Callable
from itertools import zip_longest
from numbers import Integral
from typing import Any
import numpy as np
from dask import config
from dask.array.chunk import getitem
from dask.array.core import getter, getter_inline, getter_nofancy
from dask.blockwise import fuse_roots, optimize_blockwise
from dask.core import flatten, reverse_dict
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable, fuse, inline_functions
from dask.utils import ensure_dict
def hold_keys(dsk, dependencies):
    """Find keys to avoid fusion

    We don't want to fuse data present in the graph because it is easier to
    serialize as a raw value.

    We don't want to fuse chains after getitem/GETTERS because we want to
    move around only small pieces of data, rather than the underlying arrays.
    """
    dependents = reverse_dict(dependencies)
    data = {k for k, v in dsk.items() if type(v) not in (tuple, str)}
    hold_keys = list(data)
    for dat in data:
        deps = dependents[dat]
        for dep in deps:
            task = dsk[dep]
            if _is_getter_task(task):
                try:
                    while len(dependents[dep]) == 1:
                        new_dep = next(iter(dependents[dep]))
                        new_task = dsk[new_dep]
                        if _is_getter_task(new_task) or new_task in dsk:
                            dep = new_dep
                        else:
                            break
                except (IndexError, TypeError):
                    pass
                hold_keys.append(dep)
    return hold_keys