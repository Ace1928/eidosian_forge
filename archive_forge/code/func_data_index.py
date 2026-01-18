from __future__ import annotations
from collections.abc import (
import csv as csvlib
import os
from typing import (
import numpy as np
from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import notna
from pandas.core.indexes.api import Index
from pandas.io.common import get_handle
@cache_readonly
def data_index(self) -> Index:
    data_index = self.obj.index
    if isinstance(data_index, (ABCDatetimeIndex, ABCPeriodIndex)) and self.date_format is not None:
        data_index = Index([x.strftime(self.date_format) if notna(x) else '' for x in data_index])
    elif isinstance(data_index, ABCMultiIndex):
        data_index = data_index.remove_unused_levels()
    return data_index