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
@memory.cache
def get_cancer() -> Tuple[np.ndarray, np.ndarray]:
    """Fetch the breast cancer dataset from sklearn."""
    datasets = pytest.importorskip('sklearn.datasets')
    return datasets.load_breast_cancer(return_X_y=True)