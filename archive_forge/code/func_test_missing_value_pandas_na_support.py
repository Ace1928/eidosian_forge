import warnings
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
from sklearn.utils._testing import assert_allclose, assert_array_equal
from sklearn.utils.fixes import (
@pytest.mark.parametrize('est, func', [(MaxAbsScaler(), maxabs_scale), (MinMaxScaler(), minmax_scale), (StandardScaler(), scale), (StandardScaler(with_mean=False), scale), (PowerTransformer('yeo-johnson'), power_transform), (PowerTransformer('box-cox'), power_transform), (QuantileTransformer(n_quantiles=3), quantile_transform), (RobustScaler(), robust_scale), (RobustScaler(with_centering=False), robust_scale)])
def test_missing_value_pandas_na_support(est, func):
    pd = pytest.importorskip('pandas')
    X = np.array([[1, 2, 3, np.nan, np.nan, 4, 5, 1], [np.nan, np.nan, 8, 4, 6, np.nan, np.nan, 8], [1, 2, 3, 4, 5, 6, 7, 8]]).T
    X_df = pd.DataFrame(X, dtype='Int16', columns=['a', 'b', 'c'])
    X_df['c'] = X_df['c'].astype('int')
    X_trans = est.fit_transform(X)
    X_df_trans = est.fit_transform(X_df)
    assert_allclose(X_trans, X_df_trans)