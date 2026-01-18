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
def test_feature_union_parallel():
    X = JUNK_FOOD_DOCS
    fs = FeatureUnion([('words', CountVectorizer(analyzer='word')), ('chars', CountVectorizer(analyzer='char'))])
    fs_parallel = FeatureUnion([('words', CountVectorizer(analyzer='word')), ('chars', CountVectorizer(analyzer='char'))], n_jobs=2)
    fs_parallel2 = FeatureUnion([('words', CountVectorizer(analyzer='word')), ('chars', CountVectorizer(analyzer='char'))], n_jobs=2)
    fs.fit(X)
    X_transformed = fs.transform(X)
    assert X_transformed.shape[0] == len(X)
    fs_parallel.fit(X)
    X_transformed_parallel = fs_parallel.transform(X)
    assert X_transformed.shape == X_transformed_parallel.shape
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel.toarray())
    X_transformed_parallel2 = fs_parallel2.fit_transform(X)
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel2.toarray())
    X_transformed_parallel2 = fs_parallel2.transform(X)
    assert_array_equal(X_transformed.toarray(), X_transformed_parallel2.toarray())