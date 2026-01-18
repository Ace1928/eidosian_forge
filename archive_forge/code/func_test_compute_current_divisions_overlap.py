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
def test_compute_current_divisions_overlap():
    A = pd.DataFrame({'key': [1, 2, 3, 4, 4, 5, 6, 7], 'value': list('abcd' * 2)})
    a = dd.from_pandas(A, npartitions=2)
    with pytest.warns(UserWarning, match='Partitions have overlapping values'):
        divisions = a.compute_current_divisions('key')
        b = a.set_index('key', divisions=divisions)
        assert b.divisions == (1, 4, 7)
        assert [len(p) for p in b.partitions] == [3, 5]