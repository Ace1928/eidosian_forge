from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util.version import Version
@pytest.mark.parametrize('as_index', [True, False])
@pytest.mark.parametrize('normalize, name, expected_data', [(False, 'count', np.array([2, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)), (True, 'proportion', np.array([0.5, 0.25, 0.25, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))])
def test_categorical_single_grouper_observed_false(education_df, as_index, normalize, name, expected_data, request):
    if Version(np.__version__) >= Version('1.25'):
        request.node.add_marker(pytest.mark.xfail(reason='pandas default unstable sorting of duplicatesissue with numpy>=1.25 with AVX instructions', strict=False))
    expected_index = [('FR', 'male', 'low'), ('FR', 'female', 'high'), ('FR', 'male', 'medium'), ('FR', 'female', 'low'), ('FR', 'male', 'high'), ('FR', 'female', 'medium'), ('US', 'female', 'high'), ('US', 'male', 'low'), ('US', 'male', 'medium'), ('US', 'male', 'high'), ('US', 'female', 'medium'), ('US', 'female', 'low'), ('ASIA', 'male', 'low'), ('ASIA', 'male', 'high'), ('ASIA', 'female', 'medium'), ('ASIA', 'female', 'low'), ('ASIA', 'female', 'high'), ('ASIA', 'male', 'medium')]
    assert_categorical_single_grouper(education_df=education_df, as_index=as_index, observed=False, expected_index=expected_index, normalize=normalize, name=name, expected_data=expected_data)