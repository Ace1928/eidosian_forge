from __future__ import annotations
from typing import Any
import numpy as np
import pytest
import pandas as pd
import pandas._testing as tm
def is_bool_not_implemented(data, op_name):
    return data.dtype.kind == 'b' and op_name.strip('_').lstrip('r') in ['pow', 'truediv', 'floordiv']