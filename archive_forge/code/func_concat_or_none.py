from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def concat_or_none(seq: Optional[Sequence[np.ndarray]]) -> Optional[np.ndarray]:
    """Concatenate the data if it's not None."""
    if seq:
        return concat(seq)
    return None