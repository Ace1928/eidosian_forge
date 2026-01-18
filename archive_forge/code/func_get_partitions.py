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
def get_partitions(index):
    """Grab required partitions and indices from `right` and `right_indices`."""
    must_grab = right_indices[index]
    partitions_list = np.array([right[i] for i in must_grab.keys()])
    indices_list = list(must_grab.values())
    return {'other': partitions_list, 'internal_other_indices': indices_list}