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
def test_n_features_in_feature_union():
    X = [[1, 2], [3, 4], [5, 6]]
    y = [0, 1, 2]
    ss = StandardScaler()
    fu = make_union(ss)
    assert not hasattr(fu, 'n_features_in_')
    fu.fit(X, y)
    assert fu.n_features_in_ == ss.n_features_in_ == 2
    ss = StandardScaler()
    fu = make_union(ss)
    ss.fit(X, y)
    assert fu.n_features_in_ == ss.n_features_in_ == 2