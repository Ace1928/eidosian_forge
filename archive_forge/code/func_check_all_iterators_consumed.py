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
def check_all_iterators_consumed(self):
    if len(self.iters) > 1:
        for i in self.iters:
            if isinstance(i, itertools.repeat):
                continue
            try:
                next(i)
            except StopIteration:
                pass
            else:
                msg = "map called with multiple bags that aren't identically partitioned. Please ensure that all bag arguments have the same partition lengths"
                raise ValueError(msg)