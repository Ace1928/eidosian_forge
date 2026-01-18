from typing import Callable, Optional
import numpy as np
import modin.pandas as pd
from modin.config import NPartitions
from modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe import (
from modin.core.storage_formats.pandas import PandasQueryCompiler
from modin.error_message import ErrorMessage
from modin.utils import get_current_execution
def mask_partition(df, i):
    new_length = len(df.index) // self.num_partitions
    if i == self.num_partitions - 1:
        return df.iloc[i * new_length:]
    return df.iloc[i * new_length:(i + 1) * new_length]