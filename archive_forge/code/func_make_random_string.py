from __future__ import annotations
import re
import string
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, cast
import numpy as np
import pandas as pd
from dask.dataframe._compat import PANDAS_GE_220, PANDAS_GE_300
from dask.dataframe._pyarrow import is_object_string_dtype
from dask.dataframe.core import tokenize
from dask.dataframe.io.utils import DataFrameIOFunction
from dask.utils import random_state_data
def make_random_string(n, rstate, length: int=25) -> list[str]:
    choices = list(string.ascii_letters + string.digits + string.punctuation + ' ')
    return [''.join(rstate.choice(choices, size=length)) for _ in range(n)]