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
def maybe_lift(lab, size: int) -> tuple[np.ndarray, int]:
    return (lab + 1, size + 1) if (lab == -1).any() else (lab, size)