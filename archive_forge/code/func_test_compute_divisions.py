from __future__ import annotations
import contextlib
import itertools
import multiprocessing as mp
import os
import pickle
import random
import string
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy
from datetime import date, time
from decimal import Decimal
from functools import partial
from unittest import mock
import numpy as np
import pandas as pd
import pytest
import dask
import dask.dataframe as dd
from dask.base import compute_as_if_collection
from dask.dataframe._compat import (
from dask.dataframe.shuffle import (
from dask.dataframe.utils import assert_eq, make_meta
from dask.optimization import cull
def test_compute_divisions():
    from dask.dataframe.shuffle import compute_and_set_divisions
    df = pd.DataFrame({'x': [1, 2, 3, 4], 'y': [10, 20, 20, 40], 'z': [4, 3, 2, 1]}, index=[1, 3, 10, 20])
    a = dd.from_pandas(df, 2, sort=False).clear_divisions()
    assert not a.known_divisions
    if DASK_EXPR_ENABLED:
        b = a.compute_current_divisions(set_divisions=True)
    else:
        b = compute_and_set_divisions(copy(a))
    assert_eq(a, b, check_divisions=False)
    assert b.known_divisions