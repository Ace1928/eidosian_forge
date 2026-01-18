from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import pyarrow as pa
from triad.collections.dict import IndexedOrderedDict
from triad.utils.assertion import assert_arg_not_none, assert_or_throw
from triad.utils.pandas_like import PD_UTILS
from triad.utils.pyarrow import (
from triad.utils.schema import (
def create_empty_arrow_table(self) -> pa.Table:
    """Create an empty pyarrow table based on the schema"""
    if not hasattr(pa.Table, 'from_pylist'):
        arr = [pa.array([])] * len(self)
        return pa.Table.from_arrays(arr, schema=self.pa_schema)
    return pa.Table.from_pylist([], schema=self.pa_schema)