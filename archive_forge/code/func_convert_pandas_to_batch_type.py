from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
@Deprecated
def convert_pandas_to_batch_type(data: 'pd.DataFrame', type: BatchFormat, cast_tensor_columns: bool=False):
    """Convert the provided Pandas dataframe to the provided ``type``.

    Args:
        data: A Pandas DataFrame
        type: The specific ``BatchFormat`` to convert to.
        cast_tensor_columns: Whether tensor columns should be cast to our tensor
            extension type.

    Returns:
        The input data represented with the provided type.
    """
    warnings.warn('`convert_pandas_to_batch_type` is deprecated as a developer API starting from Ray 2.4. All batch format conversions should be done manually instead of relying on this API.', PendingDeprecationWarning)
    return _convert_pandas_to_batch_type(data=data, type=type, cast_tensor_columns=cast_tensor_columns)