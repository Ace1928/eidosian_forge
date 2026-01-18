import re
from pprint import PrettyPrinter
import numpy as np
from sklearn.utils._pprint import _EstimatorPrettyPrinter
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import config_context
def test_complexity_print_changed_only():

    class DummyEstimator(TransformerMixin, BaseEstimator):
        nb_times_repr_called = 0

        def __init__(self, estimator=None):
            self.estimator = estimator

        def __repr__(self):
            DummyEstimator.nb_times_repr_called += 1
            return super().__repr__()

        def transform(self, X, copy=None):
            return X
    estimator = DummyEstimator(make_pipeline(DummyEstimator(DummyEstimator()), DummyEstimator(), 'passthrough'))
    with config_context(print_changed_only=False):
        repr(estimator)
        nb_repr_print_changed_only_false = DummyEstimator.nb_times_repr_called
    DummyEstimator.nb_times_repr_called = 0
    with config_context(print_changed_only=True):
        repr(estimator)
        nb_repr_print_changed_only_true = DummyEstimator.nb_times_repr_called
    assert nb_repr_print_changed_only_false == nb_repr_print_changed_only_true