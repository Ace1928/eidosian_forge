import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('array_1_writeable,array_2_writeable', [(True, True), (True, False), (False, False)])
def test_cut_read_only(array_1_writeable, array_2_writeable):
    array_1 = np.arange(0, 100, 10)
    array_1.flags.writeable = array_1_writeable
    array_2 = np.arange(0, 100, 10)
    array_2.flags.writeable = array_2_writeable
    hundred_elements = np.arange(100)
    tm.assert_categorical_equal(cut(hundred_elements, array_1), cut(hundred_elements, array_2))