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
def local_name(self) -> str:
    return self.name.replace(LOCAL_TAG, '')