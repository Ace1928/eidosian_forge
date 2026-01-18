from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def test_bad_subset(education_df):
    gp = education_df.groupby('country')
    with pytest.raises(ValueError, match='subset'):
        gp.value_counts(subset=['country'])