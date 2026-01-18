import math
from typing import Any, Callable, List, Optional, Tuple, TypeVar
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.dataframe.core import DataFrame
from dask.delayed import delayed
from dask.distributed import Client, get_client
from triad.utils.pandas_like import PD_UTILS, PandasLikeUtils
from triad.utils.pyarrow import to_pandas_dtype
import fugue.api as fa
from fugue.constants import FUGUE_CONF_DEFAULT_PARTITIONS
from ._constants import FUGUE_DASK_CONF_DEFAULT_PARTITIONS
def get_or_create_client(self, client: Optional[Client]=None):
    if client is not None:
        return client
    try:
        return get_client()
    except ValueError:
        return Client(processes=True)