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
def test_index_with_dataframe(shuffle_method):
    res1 = shuffle(d, d[['b']], shuffle_method=shuffle_method).compute()
    res2 = shuffle(d, ['b'], shuffle_method=shuffle_method).compute()
    res3 = shuffle(d, 'b', shuffle_method=shuffle_method).compute()
    assert sorted(res1.values.tolist()) == sorted(res2.values.tolist())
    assert sorted(res1.values.tolist()) == sorted(res3.values.tolist())