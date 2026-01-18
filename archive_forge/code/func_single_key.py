from __future__ import annotations
import operator
import types
import uuid
import warnings
from collections.abc import Sequence
from dataclasses import fields, is_dataclass, replace
from functools import partial
from tlz import concat, curry, merge, unique
from dask import config
from dask.base import (
from dask.base import tokenize as _tokenize
from dask.context import globalmethod
from dask.core import flatten, quote
from dask.highlevelgraph import HighLevelGraph
from dask.typing import Graph, NestedKeys
from dask.utils import (
def single_key(seq):
    """Pick out the only element of this list, a list of keys"""
    return seq[0]