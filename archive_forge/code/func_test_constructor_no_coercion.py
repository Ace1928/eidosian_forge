import numpy as np
import pytest
from pandas.core.dtypes.dtypes import NumpyEADtype
import pandas as pd
import pandas._testing as tm
from pandas.arrays import NumpyExtensionArray
def test_constructor_no_coercion():
    with pytest.raises(ValueError, match='NumPy array'):
        NumpyExtensionArray([1, 2, 3])