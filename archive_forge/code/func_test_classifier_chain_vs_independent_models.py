import re
import numpy as np
import pytest
from joblib import cpu_count
from sklearn import datasets
from sklearn.base import ClassifierMixin, clone
from sklearn.datasets import (
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
from sklearn.metrics import jaccard_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import (
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.utils._testing import (
from sklearn.utils.fixes import (
def test_classifier_chain_vs_independent_models():
    X, Y = generate_multilabel_dataset_with_correlations()
    X_train = X[:600, :]
    X_test = X[600:, :]
    Y_train = Y[:600, :]
    Y_test = Y[600:, :]
    ovr = OneVsRestClassifier(LogisticRegression())
    ovr.fit(X_train, Y_train)
    Y_pred_ovr = ovr.predict(X_test)
    chain = ClassifierChain(LogisticRegression())
    chain.fit(X_train, Y_train)
    Y_pred_chain = chain.predict(X_test)
    assert jaccard_score(Y_test, Y_pred_chain, average='samples') > jaccard_score(Y_test, Y_pred_ovr, average='samples')