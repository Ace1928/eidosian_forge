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
def format_array_wrap(array_, formatter_):
    fmt_values = format_array(array_, formatter_, float_format=self.float_format, na_rep=self.na_rep, digits=self.digits, space=self.space, justify=self.justify, decimal=self.decimal, leading_space=self.leading_space)
    return fmt_values