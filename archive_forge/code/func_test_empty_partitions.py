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
def test_empty_partitions():
    df = pd.DataFrame({'a': list(range(10))})
    df['b'] = df['a'] % 3
    df['c'] = df['b'].astype(str)
    ddf = dd.from_pandas(df, npartitions=3)
    ddf = ddf.set_index('b')
    ddf = ddf.repartition(npartitions=3)
    ddf.get_partition(0).compute()
    assert_eq(ddf, df.set_index('b'))
    ddf = ddf.set_index('c')
    assert_eq(ddf, df.set_index('b').set_index('c'))