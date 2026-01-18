import importlib
import inspect
import os
import warnings
from inspect import signature
from pkgutil import walk_packages
import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_classification
from sklearn.experimental import (
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import IS_PYPY, all_estimators
from sklearn.utils._testing import (
from sklearn.utils.deprecation import _is_deprecated
from sklearn.utils.estimator_checks import (
from sklearn.utils.fixes import parse_version, sp_version
@ignore_warnings(category=sklearn.exceptions.ConvergenceWarning)
@pytest.mark.filterwarnings('ignore:The SAMME.R algorithm')
@pytest.mark.parametrize('name, Estimator', all_estimators())
def test_fit_docstring_attributes(name, Estimator):
    pytest.importorskip('numpydoc')
    from numpydoc import docscrape
    doc = docscrape.ClassDoc(Estimator)
    attributes = doc['Attributes']
    if Estimator.__name__ in ('HalvingRandomSearchCV', 'RandomizedSearchCV', 'HalvingGridSearchCV', 'GridSearchCV'):
        est = _construct_searchcv_instance(Estimator)
    elif Estimator.__name__ in ('ColumnTransformer', 'Pipeline', 'FeatureUnion'):
        est = _construct_compose_pipeline_instance(Estimator)
    elif Estimator.__name__ == 'SparseCoder':
        est = _construct_sparse_coder(Estimator)
    else:
        est = _construct_instance(Estimator)
    if Estimator.__name__ == 'SelectKBest':
        est.set_params(k=2)
    elif Estimator.__name__ == 'DummyClassifier':
        est.set_params(strategy='stratified')
    elif Estimator.__name__ == 'CCA' or Estimator.__name__.startswith('PLS'):
        est.set_params(n_components=1)
    elif Estimator.__name__ in ('GaussianRandomProjection', 'SparseRandomProjection'):
        est.set_params(n_components=2)
    elif Estimator.__name__ == 'TSNE':
        est.set_params(perplexity=2)
    if Estimator.__name__ in ('LinearSVC', 'LinearSVR'):
        est.set_params(dual='auto')
    if Estimator.__name__ in ('NMF', 'MiniBatchNMF'):
        est.set_params(n_components='auto')
    if Estimator.__name__ == 'QuantileRegressor':
        solver = 'highs' if sp_version >= parse_version('1.6.0') else 'interior-point'
        est.set_params(solver=solver)
    if 'max_iter' in est.get_params():
        est.set_params(max_iter=2)
    if 'random_state' in est.get_params():
        est.set_params(random_state=0)
    skipped_attributes = {}
    if Estimator.__name__.endswith('Vectorizer'):
        if Estimator.__name__ in ('CountVectorizer', 'HashingVectorizer', 'TfidfVectorizer'):
            X = ['This is the first document.', 'This document is the second document.', 'And this is the third one.', 'Is this the first document?']
        elif Estimator.__name__ == 'DictVectorizer':
            X = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
        y = None
    else:
        X, y = make_classification(n_samples=20, n_features=3, n_redundant=0, n_classes=2, random_state=2)
        y = _enforce_estimator_tags_y(est, y)
        X = _enforce_estimator_tags_X(est, X)
    if '1dlabels' in est._get_tags()['X_types']:
        est.fit(y)
    elif '2dlabels' in est._get_tags()['X_types']:
        est.fit(np.c_[y, y])
    elif '3darray' in est._get_tags()['X_types']:
        est.fit(X[np.newaxis, ...], y)
    else:
        est.fit(X, y)
    for attr in attributes:
        if attr.name in skipped_attributes:
            continue
        desc = ' '.join(attr.desc).lower()
        if 'only ' in desc:
            continue
        with ignore_warnings(category=FutureWarning):
            assert hasattr(est, attr.name)
    fit_attr = _get_all_fitted_attributes(est)
    fit_attr_names = [attr.name for attr in attributes]
    undocumented_attrs = set(fit_attr).difference(fit_attr_names)
    undocumented_attrs = set(undocumented_attrs).difference(skipped_attributes)
    if undocumented_attrs:
        raise AssertionError(f'Undocumented attributes for {Estimator.__name__}: {undocumented_attrs}')