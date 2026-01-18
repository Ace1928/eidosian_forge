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
@pytest.mark.parametrize('module_path,expected_module', [('prefix.mymodule', 'prefix.mymodule'), ('prefix._mymodule', 'prefix'), ('prefix.mypackage._mymodule', 'prefix.mypackage'), ('prefix.mypackage._mymodule.submodule', 'prefix.mypackage'), ('prefix.mypackage.mymodule.submodule', 'prefix.mypackage.mymodule.submodule')])
def test_html_documentation_link_mixin_get_doc_link(module_path, expected_module):
    """Check the behaviour of the `_get_doc_link` with various parameter."""

    class FooBar(_HTMLDocumentationLinkMixin):
        pass
    FooBar.__module__ = module_path
    est = FooBar()
    est._doc_link_module = 'prefix'
    est._doc_link_template = 'https://website.com/{estimator_module}.{estimator_name}.html'
    assert est._get_doc_link() == f'https://website.com/{expected_module}.FooBar.html'