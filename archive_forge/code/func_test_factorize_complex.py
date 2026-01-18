from datetime import datetime
import struct
import numpy as np
import pytest
from pandas._libs import (
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
import pandas.core.algorithms as algos
from pandas.core.arrays import (
import pandas.core.common as com
def test_factorize_complex(self):
    array = [1, 2, 2 + 1j]
    msg = 'factorize with argument that is not not a Series'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        labels, uniques = algos.factorize(array)
    expected_labels = np.array([0, 1, 2], dtype=np.intp)
    tm.assert_numpy_array_equal(labels, expected_labels)
    expected_uniques = np.array([1 + 0j, 2 + 0j, 2 + 1j], dtype=object)
    tm.assert_numpy_array_equal(uniques, expected_uniques)