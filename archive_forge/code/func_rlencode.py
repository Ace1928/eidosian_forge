import os
import zipfile
from dataclasses import dataclass
from typing import Any, Generator, List, NamedTuple, Optional, Tuple, Union
from urllib import request
import numpy as np
import pytest
from numpy import typing as npt
from numpy.random import Generator as RNG
from scipy import sparse
import xgboost
from xgboost.data import pandas_pyarrow_mapper
def rlencode(x: npt.NDArray[np.int32]) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Run length encoding using numpy, modified from:
    https://gist.github.com/nvictus/66627b580c13068589957d6ab0919e66

    """
    x = np.asarray(x)
    n = x.size
    starts = np.r_[0, np.flatnonzero(~np.isclose(x[1:], x[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, n])
    values = x[starts]
    indptr = np.append(starts, np.array([x.size]))
    return (indptr, lengths, values)