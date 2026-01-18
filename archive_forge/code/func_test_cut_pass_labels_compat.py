import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_cut_pass_labels_compat():
    arr = [50, 5, 10, 15, 20, 30, 70]
    labels = ['Good', 'Medium', 'Bad']
    result = cut(arr, 3, labels=labels)
    exp = cut(arr, 3, labels=Categorical(labels, categories=labels, ordered=True))
    tm.assert_categorical_equal(result, exp)