from typing import Any
import numpy as np
from ray.air.util.tensor_extensions.utils import create_ragged_ndarray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.util import _truncated_repr
def is_valid_udf_return(udf_return_col: Any) -> bool:
    """Check whether a UDF column is valid.

    Valid columns must either be a list of elements, or an array-like object.
    """
    return isinstance(udf_return_col, list) or is_array_like(udf_return_col)