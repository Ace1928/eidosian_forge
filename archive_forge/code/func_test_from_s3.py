from __future__ import annotations
import gc
import math
import os
import random
import warnings
import weakref
from bz2 import BZ2File
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from gzip import GzipFile
from itertools import repeat
import partd
import pytest
from tlz import groupby, identity, join, merge, pluck, unique, valmap
import dask
import dask.bag as db
from dask.bag.core import (
from dask.bag.utils import assert_eq
from dask.blockwise import Blockwise
from dask.delayed import Delayed
from dask.typing import Graph
from dask.utils import filetexts, tmpdir, tmpfile
from dask.utils_test import add, hlg_layer, hlg_layer_topological, inc
@pytest.mark.xfail(reason='https://github.com/dask/dask/issues/6914')
@pytest.mark.slow
@pytest.mark.network
def test_from_s3():
    pytest.importorskip('s3fs')
    five_tips = ('total_bill,tip,sex,smoker,day,time,size\n', '16.99,1.01,Female,No,Sun,Dinner,2\n', '10.34,1.66,Male,No,Sun,Dinner,3\n', '21.01,3.5,Male,No,Sun,Dinner,3\n', '23.68,3.31,Male,No,Sun,Dinner,2\n')
    e = db.read_text('s3://tip-data/t*.gz', storage_options=dict(anon=True))
    assert e.take(5) == five_tips
    c = db.read_text(['s3://tip-data/tips.gz', 's3://tip-data/tips.json', 's3://tip-data/tips.csv'], storage_options=dict(anon=True))
    assert c.npartitions == 3