import itertools
import re
import shutil
import time
from tempfile import mkdtemp
import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils._metadata_requests import COMPOSITE_METHODS, METHODS
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
from sklearn.utils.validation import check_is_fitted
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_feature_union(csr_container):
    X = iris.data
    X -= X.mean(axis=0)
    y = iris.target
    svd = TruncatedSVD(n_components=2, random_state=0)
    select = SelectKBest(k=1)
    fs = FeatureUnion([('svd', svd), ('select', select)])
    fs.fit(X, y)
    X_transformed = fs.transform(X)
    assert X_transformed.shape == (X.shape[0], 3)
    assert_array_almost_equal(X_transformed[:, :-1], svd.fit_transform(X))
    assert_array_equal(X_transformed[:, -1], select.fit_transform(X, y).ravel())
    fs = FeatureUnion([('svd', svd), ('select', select)])
    X_sp = csr_container(X)
    X_sp_transformed = fs.fit_transform(X_sp, y)
    assert_array_almost_equal(X_transformed, X_sp_transformed.toarray())
    fs2 = clone(fs)
    assert fs.transformer_list[0][1] is not fs2.transformer_list[0][1]
    fs.set_params(select__k=2)
    assert fs.fit_transform(X, y).shape == (X.shape[0], 4)
    fs = FeatureUnion([('mock', Transf()), ('svd', svd), ('select', select)])
    X_transformed = fs.fit_transform(X, y)
    assert X_transformed.shape == (X.shape[0], 8)
    msg = 'All estimators should implement fit and transform.*\\bNoTrans\\b'
    fs = FeatureUnion([('transform', Transf()), ('no_transform', NoTrans())])
    with pytest.raises(TypeError, match=msg):
        fs.fit(X)
    fs = FeatureUnion((('svd', svd), ('select', select)))
    fs.fit(X, y)