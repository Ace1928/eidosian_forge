import numpy as np
import pytest
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm._bounds import l1_min_c
from sklearn.svm._newrand import bounded_rand_int_wrap, set_seed_wrap
from sklearn.utils.fixes import CSR_CONTAINERS
def check_l1_min_c(X, y, loss, fit_intercept=True, intercept_scaling=1.0):
    min_c = l1_min_c(X, y, loss=loss, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling)
    clf = {'log': LogisticRegression(penalty='l1', solver='liblinear'), 'squared_hinge': LinearSVC(loss='squared_hinge', penalty='l1', dual=False)}[loss]
    clf.fit_intercept = fit_intercept
    clf.intercept_scaling = intercept_scaling
    clf.C = min_c
    clf.fit(X, y)
    assert (np.asarray(clf.coef_) == 0).all()
    assert (np.asarray(clf.intercept_) == 0).all()
    clf.C = min_c * 1.01
    clf.fit(X, y)
    assert (np.asarray(clf.coef_) != 0).any() or (np.asarray(clf.intercept_) != 0).any()