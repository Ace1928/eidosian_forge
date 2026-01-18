from __future__ import annotations
from typing import Any
import numpy as np
from pandas._libs.lib import infer_dtype
from pandas._libs.tslibs import iNaT
from pandas.errors import NoBufferPresent
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.dtypes import (
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.core.interchange.buffer import PandasBuffer
from pandas.core.interchange.dataframe_protocol import (
from pandas.core.interchange.utils import (
def num_chunks(self) -> int:
    """
        Return the number of chunks the column consists of.
        """
    return 1