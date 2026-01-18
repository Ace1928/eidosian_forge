from __future__ import annotations
import re
from typing import (
import warnings
import numpy as np
from pandas._typing import (
from pandas.errors import PerformanceWarning
from pandas.util._exceptions import find_stack_level
from pandas.core.dtypes.astype import astype_array
from pandas.core.dtypes.base import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.missing import (
@property
def subtype(self):
    return self._dtype