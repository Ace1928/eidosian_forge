import numbers
import os
from packaging.version import Version
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas._typing import Dtype
from pandas.compat import set_function_name
from pandas.core.dtypes.generic import ABCDataFrame, ABCSeries
from pandas.core.indexers import check_array_indexer, validate_indices
from pandas.io.formats.format import ExtensionArrayFormatter
from ray.air.util.tensor_extensions.utils import (
from ray.util.annotations import PublicAPI
@property
def is_variable_shaped(self):
    """
        Whether this TensorArray holds variable-shaped tensor elements.
        """
    if self._is_variable_shaped is None:
        self._is_variable_shaped = _is_ndarray_variable_shaped_tensor(self._tensor)
    return self._is_variable_shaped