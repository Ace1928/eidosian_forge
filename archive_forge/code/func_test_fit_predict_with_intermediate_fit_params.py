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
def test_fit_predict_with_intermediate_fit_params():
    pipe = Pipeline([('transf', TransfFitParams()), ('clf', FitParamT())])
    pipe.fit_predict(X=None, y=None, transf__should_get_this=True, clf__should_succeed=True)
    assert pipe.named_steps['transf'].fit_params['should_get_this']
    assert pipe.named_steps['clf'].successful
    assert 'should_succeed' not in pipe.named_steps['transf'].fit_params