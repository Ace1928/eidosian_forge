import os
import warnings
from abc import ABC
from functools import wraps
from typing import TYPE_CHECKING
import numpy as np
import pandas
from pandas._libs.lib import no_default
from modin.config import (
from modin.core.dataframe.pandas.utils import create_pandas_df_from_partitions
from modin.core.storage_formats.pandas.utils import compute_chunksize
from modin.error_message import ErrorMessage
from modin.logging import ClassLogger
def compute_part_size(indexer, remote_part, part_idx, axis):
    """Compute indexer length along the specified axis for the passed partition."""
    if isinstance(indexer, slice):
        shapes_container = row_lengths if axis == 0 else col_widths
        part_size = shapes_container[part_idx]
        if part_size is None:
            part_size = remote_part.length() if axis == 0 else remote_part.width()
            shapes_container[part_idx] = part_size
        indexer = range(*indexer.indices(part_size))
    return len(indexer)