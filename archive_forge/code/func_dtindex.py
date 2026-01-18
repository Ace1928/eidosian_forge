import numpy as np
import pytest
from pandas._libs.tslibs import fields
import pandas._testing as tm
@pytest.fixture
def dtindex():
    dtindex = np.arange(5, dtype=np.int64) * 10 ** 9 * 3600 * 24 * 32
    dtindex.flags.writeable = False
    return dtindex