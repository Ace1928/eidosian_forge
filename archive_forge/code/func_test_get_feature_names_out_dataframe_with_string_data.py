import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('feature_names_out, expected', [('one-to-one', ['pet', 'color']), [lambda est, names: [f'{n}_out' for n in names], ['pet_out', 'color_out']]])
@pytest.mark.parametrize('in_pipeline', [True, False])
def test_get_feature_names_out_dataframe_with_string_data(feature_names_out, expected, in_pipeline):
    """Check that get_feature_names_out works with DataFrames with string data."""
    pd = pytest.importorskip('pandas')
    X = pd.DataFrame({'pet': ['dog', 'cat'], 'color': ['red', 'green']})

    def func(X):
        if feature_names_out == 'one-to-one':
            return X
        else:
            name = feature_names_out(None, X.columns)
            return X.rename(columns=dict(zip(X.columns, name)))
    transformer = FunctionTransformer(func=func, feature_names_out=feature_names_out)
    if in_pipeline:
        transformer = make_pipeline(transformer)
    X_trans = transformer.fit_transform(X)
    assert isinstance(X_trans, pd.DataFrame)
    names = transformer.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, expected)