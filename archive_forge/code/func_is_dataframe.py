from __future__ import annotations
import datetime as dt
import os
import sys
from typing import Any, Iterable
import numpy as np
import param
def is_dataframe(obj) -> bool:
    if 'pandas' not in sys.modules:
        return False
    import pandas as pd
    return isinstance(obj, pd.DataFrame)