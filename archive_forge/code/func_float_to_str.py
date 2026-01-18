from typing import Dict
from ml_dtypes._custom_floats import bfloat16
from ml_dtypes._custom_floats import float8_e4m3b11fnuz
from ml_dtypes._custom_floats import float8_e4m3fn
from ml_dtypes._custom_floats import float8_e4m3fnuz
from ml_dtypes._custom_floats import float8_e5m2
from ml_dtypes._custom_floats import float8_e5m2fnuz
import numpy as np
def float_to_str(f):
    return '%6.2e' % float(f)