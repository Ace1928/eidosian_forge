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
def get_ames_housing() -> Tuple[np.ndarray, np.ndarray]:
    """
    Number of samples: 1460
    Number of features: 20
    Number of categorical features: 10
    Number of numerical features: 10
    """
    datasets = pytest.importorskip('sklearn.datasets')
    X, y = datasets.fetch_openml(data_id=42165, as_frame=True, return_X_y=True)
    categorical_columns_subset: List[str] = ['BldgType', 'GarageFinish', 'LotConfig', 'Functional', 'MasVnrType', 'HouseStyle', 'FireplaceQu', 'ExterCond', 'ExterQual', 'PoolQC']
    numerical_columns_subset: List[str] = ['3SsnPorch', 'Fireplaces', 'BsmtHalfBath', 'HalfBath', 'GarageCars', 'TotRmsAbvGrd', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 'ScreenPorch']
    X = X[categorical_columns_subset + numerical_columns_subset]
    X[categorical_columns_subset] = X[categorical_columns_subset].astype('category')
    return (X, y)