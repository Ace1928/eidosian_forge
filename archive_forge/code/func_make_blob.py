from collections import defaultdict, namedtuple
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from xgboost import DataIter, DMatrix, QuantileDMatrix, XGBModel
from xgboost.compat import concat
from .._typing import ArrayLike
from .utils import get_logger  # type: ignore
def make_blob(part: pd.DataFrame, is_valid: bool) -> None:
    append(part, alias.data, is_valid)
    append(part, alias.label, is_valid)
    append(part, alias.weight, is_valid)
    append(part, alias.margin, is_valid)
    append(part, alias.qid, is_valid)