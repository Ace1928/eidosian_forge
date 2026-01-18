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
def test_maybe_buffered_partd(tmp_path):
    import partd
    f = maybe_buffered_partd()
    p1 = f()
    assert isinstance(p1.partd, partd.Buffer)
    f2 = pickle.loads(pickle.dumps(f))
    assert not f2.buffer
    p2 = f2()
    assert isinstance(p2.partd, partd.File)
    f3 = maybe_buffered_partd(tempdir=tmp_path)
    p3 = f3()
    assert isinstance(p3.partd, partd.Buffer)
    contents = list(tmp_path.iterdir())
    assert len(contents) == 1
    assert contents[0].suffix == '.partd'
    assert contents[0].parent == tmp_path
    f4 = pickle.loads(pickle.dumps(f3))
    assert not f4.buffer
    assert f4.tempdir == tmp_path