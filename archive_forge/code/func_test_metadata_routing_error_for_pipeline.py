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
@pytest.mark.usefixtures('enable_slep006')
@pytest.mark.parametrize('method', sorted(set(METHODS) - {'split', 'partial_fit'}))
def test_metadata_routing_error_for_pipeline(method):
    """Test that metadata is not routed for pipelines when not requested."""
    X, y = ([[1]], [1])
    sample_weight, prop = ([1], 'a')
    est = SimpleEstimator()
    pipeline = Pipeline([('estimator', est)])
    error_message = f'[sample_weight, prop] are passed but are not explicitly set as requested or not for SimpleEstimator.{method}'
    with pytest.raises(ValueError, match=re.escape(error_message)):
        try:
            getattr(pipeline, method)(X, y, sample_weight=sample_weight, prop=prop)
        except TypeError:
            getattr(pipeline, method)(X, sample_weight=sample_weight, prop=prop)