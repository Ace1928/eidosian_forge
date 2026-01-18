import warnings
import numpy as np
import pytest
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.preprocessing._function_transformer import _get_adapter_from_container
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
def test_function_transformer_feature_names_out_uses_estimator():

    def add_n_random_features(X, n):
        return np.concatenate([X, np.random.rand(len(X), n)], axis=1)

    def feature_names_out(transformer, input_features):
        n = transformer.kw_args['n']
        return list(input_features) + [f'rnd{i}' for i in range(n)]
    transformer = FunctionTransformer(func=add_n_random_features, feature_names_out=feature_names_out, kw_args=dict(n=3), validate=True)
    pd = pytest.importorskip('pandas')
    df = pd.DataFrame({'a': np.random.rand(100), 'b': np.random.rand(100)})
    transformer.fit_transform(df)
    names = transformer.get_feature_names_out()
    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert_array_equal(names, ('a', 'b', 'rnd0', 'rnd1', 'rnd2'))