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
def random_state_data_python(n: int, random_state: int | Random | None=None) -> list[tuple[int, tuple[int, ...], None]]:
    """Return a list of tuples that can be passed to
    ``random.Random.setstate``.

    Parameters
    ----------
    n : int
        Number of tuples to return.
    random_state : int or ``random.Random``, optional
        If an int, is used to seed a new ``random.Random``.

    See Also
    --------
    dask.utils.random_state_data
    """
    maxuint32 = 1 << 32
    try:
        import numpy as np
        if isinstance(random_state, Random):
            random_state = random_state.randint(0, maxuint32)
        np_rng = np.random.default_rng(random_state)
        random_data = np_rng.bytes(624 * n * 4)
        arr = np.frombuffer(random_data, dtype=np.uint32).reshape((n, -1))
        return [(3, tuple(row) + (624,), None) for row in arr.tolist()]
    except ImportError:
        if not isinstance(random_state, Random):
            random_state = Random(random_state)
        return [(3, tuple((random_state.randint(0, maxuint32) for _ in range(624))) + (624,), None) for _ in range(n)]