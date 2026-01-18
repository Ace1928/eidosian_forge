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
def format_strings_slim(array_, leading_space):
    formatter = partial(pprint_thing, escape_chars=('\t', '\r', '\n'))

    def _format(x):
        return str(formatter(x))
    fmt_values = []
    for v in array_:
        tpl = '{v}' if leading_space is False else ' {v}'
        fmt_values.append(tpl.format(v=_format(v)))
    return fmt_values