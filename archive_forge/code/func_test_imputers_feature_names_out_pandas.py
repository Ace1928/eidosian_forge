import numpy as np
import pytest
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
@pytest.mark.parametrize('imputer', imputers(), ids=lambda x: x.__class__.__name__)
@pytest.mark.parametrize('add_indicator', [True, False])
def test_imputers_feature_names_out_pandas(imputer, add_indicator):
    """Check feature names out for imputers."""
    pd = pytest.importorskip('pandas')
    marker = np.nan
    imputer = imputer.set_params(add_indicator=add_indicator, missing_values=marker)
    X = np.array([[marker, 1, 5, 3, marker, 1], [2, marker, 1, 4, marker, 2], [6, 3, 7, marker, marker, 3], [1, 2, 9, 8, marker, 4]])
    X_df = pd.DataFrame(X, columns=['a', 'b', 'c', 'd', 'e', 'f'])
    imputer.fit(X_df)
    names = imputer.get_feature_names_out()
    if add_indicator:
        expected_names = ['a', 'b', 'c', 'd', 'f', 'missingindicator_a', 'missingindicator_b', 'missingindicator_d', 'missingindicator_e']
        assert_array_equal(expected_names, names)
    else:
        expected_names = ['a', 'b', 'c', 'd', 'f']
        assert_array_equal(expected_names, names)