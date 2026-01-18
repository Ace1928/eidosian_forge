import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
@pytest.mark.parametrize('get_labels,get_expected', [(lambda labels: labels, lambda labels: Categorical(['Medium'] + 4 * ['Small'] + ['Medium', 'Large'], categories=labels, ordered=True)), (lambda labels: Categorical.from_codes([0, 1, 2], labels), lambda labels: Categorical.from_codes([1] + 4 * [0] + [1, 2], labels))])
def test_cut_pass_labels(get_labels, get_expected):
    bins = [0, 25, 50, 100]
    arr = [50, 5, 10, 15, 20, 30, 70]
    labels = ['Small', 'Medium', 'Large']
    result = cut(arr, bins, labels=get_labels(labels))
    tm.assert_categorical_equal(result, get_expected(labels))