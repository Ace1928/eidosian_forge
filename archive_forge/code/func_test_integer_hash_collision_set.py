from datetime import (
import pickle
import numpy as np
import pytest
from pandas._libs.missing import NA
from pandas.core.dtypes.common import is_scalar
import pandas as pd
import pandas._testing as tm
def test_integer_hash_collision_set():
    result = {NA, hash(NA)}
    assert len(result) == 2
    assert NA in result
    assert hash(NA) in result