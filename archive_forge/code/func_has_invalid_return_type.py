from __future__ import annotations
from datetime import datetime
from functools import partial
import operator
from typing import (
import numpy as np
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
import pandas.core.common as com
from pandas.core.computation.common import (
from pandas.core.computation.scope import DEFAULT_GLOBALS
from pandas.io.formats.printing import (
@property
def has_invalid_return_type(self) -> bool:
    types = self.operand_types
    obj_dtype_set = frozenset([np.dtype('object')])
    return self.return_type == object and types - obj_dtype_set