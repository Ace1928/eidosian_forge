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
def assert_not_empty(self) -> 'Schema':
    """Raise exception if schema is empty"""
    if len(self) > 0:
        return self
    raise SchemaError("Schema can't be empty")