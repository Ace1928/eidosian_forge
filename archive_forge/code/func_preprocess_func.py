import warnings
import numpy as np
import pandas
from modin.config import NPartitions
from modin.core.io import SQLDispatcher
@classmethod
def preprocess_func(cls):
    """Prepare a function for transmission to remote workers."""
    if cls.__read_sql_with_offset is None:
        from modin.experimental.core.io.sql.utils import read_sql_with_offset
        cls.__read_sql_with_offset = cls.put(read_sql_with_offset)
    return cls.__read_sql_with_offset