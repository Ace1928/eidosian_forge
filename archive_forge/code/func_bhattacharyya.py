from __future__ import annotations
import math
import random as rnd
import pytest
import dask.bag as db
from dask.bag import random
def bhattacharyya(h1, h2):
    return 1 - sum((math.sqrt(hi * hj) for hi, hj in zip(h1, h2)))