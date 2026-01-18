from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('as_index', [False, True])
@pytest.mark.parametrize('observed', [False, True])
@pytest.mark.parametrize('normalize, name, expected_data', [(False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0], dtype=np.int64)), (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0]))])
def test_categorical_non_groupers(education_df, as_index, observed, normalize, name, expected_data, request):
    if Version(np.__version__) >= Version('1.25'):
        request.node.add_marker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
    education_df = education_df.copy()
    education_df['gender'] = education_df['gender'].astype('category')
    education_df['education'] = education_df['education'].astype('category')
    gp = education_df.groupby('country', as_index=as_index, observed=observed)
    result = gp.value_counts(normalize=normalize)
    expected_index = [('FR', 'male', 'low'), ('FR', 'female', 'high'), ('FR', 'male', 'medium'), ('FR', 'female', 'low'), ('FR', 'female', 'medium'), ('FR', 'male', 'high'), ('US', 'female', 'high'), ('US', 'male', 'low'), ('US', 'female', 'low'), ('US', 'female', 'medium'), ('US', 'male', 'high'), ('US', 'male', 'medium')]
    expected_series = Series(data=expected_data, index=MultiIndex.from_tuples(expected_index, names=['country', 'gender', 'education']), name=name)
    for i in range(1, 3):
        expected_series.index = expected_series.index.set_levels(CategoricalIndex(expected_series.index.levels[i]), level=i)
    if as_index:
        tm.assert_series_equal(result, expected_series)
    else:
        expected = expected_series.reset_index(name='proportion' if normalize else 'count')
        tm.assert_frame_equal(result, expected)