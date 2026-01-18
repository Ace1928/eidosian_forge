from __future__ import annotations
from collections import defaultdict
from typing import (
import numpy as np
from pandas._libs import (
from pandas._libs.hashtable import unique_label_indices
from pandas.core.dtypes.common import (
from pandas.core.dtypes.generic import (
from pandas.core.dtypes.missing import isna
from pandas.core.construction import extract_array
def is_int64_overflow_possible(shape: Shape) -> bool:
    the_prod = 1
    for x in shape:
        the_prod *= int(x)
    return the_prod >= lib.i8max