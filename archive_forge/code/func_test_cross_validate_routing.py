import os
import re
import sys
import tempfile
import warnings
from functools import partial
from io import StringIO
from time import sleep
import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans
from sklearn.datasets import (
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import FitFailedWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import (
from sklearn.model_selection import (
from sklearn.model_selection._validation import (
from sklearn.model_selection.tests.common import OneTimeSplitter
from sklearn.model_selection.tests.test_search import FailingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC, LinearSVC
from sklearn.tests.metadata_routing_common import (
from sklearn.utils import shuffle
from sklearn.utils._mocking import CheckingClassifier, MockDataFrame
from sklearn.utils._testing import (
from sklearn.utils.fixes import COO_CONTAINERS, CSR_CONTAINERS
from sklearn.utils.validation import _num_samples
@pytest.mark.usefixtures('enable_slep006')
@pytest.mark.parametrize('cv_method', [cross_validate, cross_val_score, cross_val_predict])
def test_cross_validate_routing(cv_method):
    """Check that the respective cv method is properly dispatching the metadata
    to the consumer."""
    scorer_registry = _Registry()
    scorer = ConsumingScorer(registry=scorer_registry).set_score_request(sample_weight='score_weights', metadata='score_metadata')
    splitter_registry = _Registry()
    splitter = ConsumingSplitter(registry=splitter_registry).set_split_request(groups='split_groups', metadata='split_metadata')
    estimator_registry = _Registry()
    estimator = ConsumingClassifier(registry=estimator_registry).set_fit_request(sample_weight='fit_sample_weight', metadata='fit_metadata')
    n_samples = _num_samples(X)
    rng = np.random.RandomState(0)
    score_weights = rng.rand(n_samples)
    score_metadata = rng.rand(n_samples)
    split_groups = rng.randint(0, 3, n_samples)
    split_metadata = rng.rand(n_samples)
    fit_sample_weight = rng.rand(n_samples)
    fit_metadata = rng.rand(n_samples)
    extra_params = {cross_validate: dict(scoring=dict(my_scorer=scorer, accuracy='accuracy')), cross_val_score: dict(scoring=scorer), cross_val_predict: dict()}
    params = dict(split_groups=split_groups, split_metadata=split_metadata, fit_sample_weight=fit_sample_weight, fit_metadata=fit_metadata)
    if cv_method is not cross_val_predict:
        params.update(score_weights=score_weights, score_metadata=score_metadata)
    cv_method(estimator, X=X, y=y, cv=splitter, **extra_params[cv_method], params=params)
    if cv_method is not cross_val_predict:
        assert len(scorer_registry)
    for _scorer in scorer_registry:
        check_recorded_metadata(obj=_scorer, method='score', split_params=('sample_weight', 'metadata'), sample_weight=score_weights, metadata=score_metadata)
    assert len(splitter_registry)
    for _splitter in splitter_registry:
        check_recorded_metadata(obj=_splitter, method='split', groups=split_groups, metadata=split_metadata)
    assert len(estimator_registry)
    for _estimator in estimator_registry:
        check_recorded_metadata(obj=_estimator, method='fit', split_params=('sample_weight', 'metadata'), sample_weight=fit_sample_weight, metadata=fit_metadata)