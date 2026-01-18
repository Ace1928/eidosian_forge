from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
def assert_categorical_single_grouper(education_df, as_index, observed, expected_index, normalize, name, expected_data):
    education_df = education_df.copy().astype('category')
    education_df['country'] = education_df['country'].cat.add_categories(['ASIA'])
    gp = education_df.groupby('country', as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)
    expected_series = Series(data=expected_data, index=MultiIndex.from_tuples(expected_index, names=['country', 'gender', 'education']), name=name)
    for i in range(3):
        index_level = CategoricalIndex(expected_series.index.levels[i])
        if i == 0:
            index_level = index_level.set_categories(education_df['country'].cat.categories)
        expected_series.index = expected_series.index.set_levels(index_level, level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name=name)
        tm.assert_frame_equal(result, expected)