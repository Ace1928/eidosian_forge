from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def cache_partitions(iterator: Iterator[pd.DataFrame], append: Callable[[pd.DataFrame, str, bool], None]) -> None:
    """Extract partitions from pyspark iterator. `append` is a user defined function for
    accepting new partition."""

    def make_blob(part: pd.DataFrame, is_valid: bool) -> None:
        append(part, alias.data, is_valid)
        append(part, alias.label, is_valid)
        append(part, alias.weight, is_valid)
        append(part, alias.margin, is_valid)
        append(part, alias.qid, is_valid)
    has_validation: Optional[bool] = None
    for part in iterator:
        if has_validation is None:
            has_validation = alias.valid in part.columns
        if has_validation is True:
            assert alias.valid in part.columns
        if has_validation:
            train = part.loc[~part[alias.valid], :]
            valid = part.loc[part[alias.valid], :]
        else:
            train, valid = (part, None)
        make_blob(train, False)
        if valid is not None:
            make_blob(valid, True)