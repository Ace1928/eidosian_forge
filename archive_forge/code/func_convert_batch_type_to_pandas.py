from enum import Enum
from typing import Dict, Union, List, TYPE_CHECKING
import warnings
import numpy as np
from ray.air.data_batch_type import DataBatchType
from ray.air.constants import TENSOR_COLUMN_NAME
from ray.util.annotations import Deprecated, DeveloperAPI
@Deprecated
def convert_batch_type_to_pandas(data: DataBatchType, cast_tensor_columns: bool=False):
    """Convert the provided data to a Pandas DataFrame.

    This API is deprecated from Ray 2.4.

    Args:
        data: Data of type DataBatchType
        cast_tensor_columns: Whether tensor columns should be cast to NumPy ndarrays.

    Returns:
        A pandas Dataframe representation of the input data.

    """
    warnings.warn('`convert_batch_type_to_pandas` is deprecated as a developer API starting from Ray 2.4. All batch format conversions should be done manually instead of relying on this API.', PendingDeprecationWarning)
    return _convert_batch_type_to_pandas(data=data, cast_tensor_columns=cast_tensor_columns)