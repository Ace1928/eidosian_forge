import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_with_timestamp_tuple_labels():
    labels = [(Timestamp(10),), (Timestamp(20),), (Timestamp(30),)]
    result = cut([2, 4, 6], bins=[1, 3, 5, 7], labels=labels)
    expected = Categorical.from_codes([0, 1, 2], labels, ordered=True)
    tm.assert_categorical_equal(result, expected)