import atexit
import os
import unittest
import warnings
import numpy as np
import pytest
from scipy import sparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import _IS_WASM
from sklearn.utils._testing import (
from sklearn.utils.deprecation import deprecated
from sklearn.utils.fixes import (
from sklearn.utils.metaestimators import available_if
def test_ignore_warning():

    def _warning_function():
        warnings.warn('deprecation warning', DeprecationWarning)

    def _multiple_warning_function():
        warnings.warn('deprecation warning', DeprecationWarning)
        warnings.warn('deprecation warning')
    assert_no_warnings(ignore_warnings(_warning_function))
    assert_no_warnings(ignore_warnings(_warning_function, category=DeprecationWarning))
    with pytest.warns(DeprecationWarning):
        ignore_warnings(_warning_function, category=UserWarning)()
    with pytest.warns() as record:
        ignore_warnings(_multiple_warning_function, category=FutureWarning)()
    assert len(record) == 2
    assert isinstance(record[0].message, DeprecationWarning)
    assert isinstance(record[1].message, UserWarning)
    with pytest.warns() as record:
        ignore_warnings(_multiple_warning_function, category=UserWarning)()
    assert len(record) == 1
    assert isinstance(record[0].message, DeprecationWarning)
    assert_no_warnings(ignore_warnings(_warning_function, category=(DeprecationWarning, UserWarning)))

    @ignore_warnings
    def decorator_no_warning():
        _warning_function()
        _multiple_warning_function()

    @ignore_warnings(category=(DeprecationWarning, UserWarning))
    def decorator_no_warning_multiple():
        _multiple_warning_function()

    @ignore_warnings(category=DeprecationWarning)
    def decorator_no_deprecation_warning():
        _warning_function()

    @ignore_warnings(category=UserWarning)
    def decorator_no_user_warning():
        _warning_function()

    @ignore_warnings(category=DeprecationWarning)
    def decorator_no_deprecation_multiple_warning():
        _multiple_warning_function()

    @ignore_warnings(category=UserWarning)
    def decorator_no_user_multiple_warning():
        _multiple_warning_function()
    assert_no_warnings(decorator_no_warning)
    assert_no_warnings(decorator_no_warning_multiple)
    assert_no_warnings(decorator_no_deprecation_warning)
    with pytest.warns(DeprecationWarning):
        decorator_no_user_warning()
    with pytest.warns(UserWarning):
        decorator_no_deprecation_multiple_warning()
    with pytest.warns(DeprecationWarning):
        decorator_no_user_multiple_warning()

    def context_manager_no_warning():
        with ignore_warnings():
            _warning_function()

    def context_manager_no_warning_multiple():
        with ignore_warnings(category=(DeprecationWarning, UserWarning)):
            _multiple_warning_function()

    def context_manager_no_deprecation_warning():
        with ignore_warnings(category=DeprecationWarning):
            _warning_function()

    def context_manager_no_user_warning():
        with ignore_warnings(category=UserWarning):
            _warning_function()

    def context_manager_no_deprecation_multiple_warning():
        with ignore_warnings(category=DeprecationWarning):
            _multiple_warning_function()

    def context_manager_no_user_multiple_warning():
        with ignore_warnings(category=UserWarning):
            _multiple_warning_function()
    assert_no_warnings(context_manager_no_warning)
    assert_no_warnings(context_manager_no_warning_multiple)
    assert_no_warnings(context_manager_no_deprecation_warning)
    with pytest.warns(DeprecationWarning):
        context_manager_no_user_warning()
    with pytest.warns(UserWarning):
        context_manager_no_deprecation_multiple_warning()
    with pytest.warns(DeprecationWarning):
        context_manager_no_user_multiple_warning()
    warning_class = UserWarning
    match = "'obj' should be a callable.+you should use 'category=UserWarning'"
    with pytest.raises(ValueError, match=match):
        silence_warnings_func = ignore_warnings(warning_class)(_warning_function)
        silence_warnings_func()
    with pytest.raises(ValueError, match=match):

        @ignore_warnings(warning_class)
        def test():
            pass