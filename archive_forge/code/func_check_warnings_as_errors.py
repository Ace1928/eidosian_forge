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
def check_warnings_as_errors(warning_info, warnings_as_errors):
    if warning_info.action == 'error' and warnings_as_errors:
        with pytest.raises(warning_info.category, match=warning_info.message):
            warnings.warn(message=warning_info.message, category=warning_info.category)
    if warning_info.action == 'ignore':
        with warnings.catch_warnings(record=True) as record:
            message = warning_info.message
            if 'Pyarrow' in message:
                message = '\nPyarrow will become a required dependency'
            warnings.warn(message=message, category=warning_info.category)
            assert len(record) == 0 if warnings_as_errors else 1
            if record:
                assert str(record[0].message) == message
                assert record[0].category == warning_info.category