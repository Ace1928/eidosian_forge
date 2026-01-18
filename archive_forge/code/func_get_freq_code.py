import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.period import (
import pandas._testing as tm
def get_freq_code(freqstr: str) -> int:
    off = to_offset(freqstr)
    code = off._period_dtype_code
    return code