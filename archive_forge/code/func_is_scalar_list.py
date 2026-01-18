from typing import Any
import numpy as np
from ray.air.util.tensor_extensions.utils import create_ragged_ndarray
from ray.data._internal.dataset_logger import DatasetLogger
from ray.data._internal.util import _truncated_repr
def is_scalar_list(udf_return_col: Any) -> bool:
    """Check whether a UDF column is is a scalar list."""
    return isinstance(udf_return_col, list) and (not udf_return_col or np.isscalar(udf_return_col[0]))