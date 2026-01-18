import html
import locale
import re
from contextlib import closing
from io import StringIO
from unittest.mock import patch
import pytest
from sklearn import config_context
from sklearn.base import BaseEstimator
from sklearn.cluster import AgglomerativeClustering, Birch
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import StackingClassifier, StackingRegressor, VotingClassifier
from sklearn.feature_selection import SelectPercentile
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._estimator_html_repr import (
from sklearn.utils.fixes import parse_version
def test_get_visual_block_pipeline():
    pipe = Pipeline([('imputer', SimpleImputer()), ('do_nothing', 'passthrough'), ('do_nothing_more', None), ('classifier', LogisticRegression())])
    est_html_info = _get_visual_block(pipe)
    assert est_html_info.kind == 'serial'
    assert est_html_info.estimators == tuple((step[1] for step in pipe.steps))
    assert est_html_info.names == ['imputer: SimpleImputer', 'do_nothing: passthrough', 'do_nothing_more: passthrough', 'classifier: LogisticRegression']
    assert est_html_info.name_details == [str(est) for _, est in pipe.steps]