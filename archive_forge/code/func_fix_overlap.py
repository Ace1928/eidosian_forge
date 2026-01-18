from __future__ import annotations
import contextlib
import logging
import math
import shutil
import tempfile
import uuid
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal
import numpy as np
import pandas as pd
import tlz as toolz
from pandas.api.types import is_numeric_dtype
from dask import config
from dask.base import compute, compute_as_if_collection, is_dask_collection, tokenize
from dask.dataframe import methods
from dask.dataframe._compat import PANDAS_GE_300
from dask.dataframe.core import (
from dask.dataframe.dispatch import (
from dask.dataframe.utils import UNKNOWN_CATEGORIES
from dask.highlevelgraph import HighLevelGraph
from dask.layers import ShuffleLayer, SimpleShuffleLayer
from dask.sizeof import sizeof
from dask.utils import M, digit, get_default_shuffle_method
def fix_overlap(ddf, mins, maxes, lens):
    """Ensures that the upper bound on each partition of ddf (except the last) is exclusive

    This is accomplished by first removing empty partitions, then altering existing
    partitions as needed to include all the values for a particular index value in
    one partition.
    """
    name = 'fix-overlap-' + tokenize(ddf, mins, maxes, lens)
    non_empties = [i for i, length in enumerate(lens) if length != 0]
    if len(non_empties) == 0:
        divisions = (None, None)
        dsk = {(name, 0): (ddf._name, 0)}
        graph = HighLevelGraph.from_collections(name, dsk, dependencies=[ddf])
        return new_dd_object(graph, name, ddf._meta, divisions)
    dsk = {(name, i): (ddf._name, div) for i, div in enumerate(non_empties)}
    ddf_keys = list(dsk.values())
    divisions = tuple(mins) + (maxes[-1],)
    overlap = [i for i in range(1, len(mins)) if mins[i] >= maxes[i - 1]]
    frames = []
    for i in overlap:
        frames.append((get_overlap, ddf_keys[i - 1], divisions[i]))
        dsk[name, i - 1] = (drop_overlap, dsk[name, i - 1], divisions[i])
        if divisions[i] == divisions[i + 1] and i + 1 in overlap:
            continue
        frames.append(ddf_keys[i])
        dsk[name, i] = (methods.concat, frames)
        frames = []
    graph = HighLevelGraph.from_collections(name, dsk, dependencies=[ddf])
    return new_dd_object(graph, name, ddf._meta, divisions)