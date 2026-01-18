from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def stack_series(series: pd.Series) -> np.ndarray:
    """Stack a series of arrays."""
    array = series.to_numpy(copy=False)
    array = np.stack(array)
    return array