import pickle
import re
import warnings
from collections import defaultdict
from collections.abc import Mapping
from functools import partial
from io import StringIO
from itertools import product
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from scipy import sparse
from sklearn.base import clone
from sklearn.feature_extraction.text import (
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.utils import _IS_WASM, IS_PYPY
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSC_CONTAINERS, CSR_CONTAINERS
@pytest.mark.parametrize('Estimator', [CountVectorizer, TfidfVectorizer, pytest.param(HashingVectorizer, marks=fails_if_pypy)])
@pytest.mark.parametrize('analyzer', [lambda doc: open(doc, 'r'), lambda doc: doc.read()])
@pytest.mark.parametrize('input_type', ['file', 'filename'])
def test_callable_analyzer_change_behavior(Estimator, analyzer, input_type):
    data = ['this is text, not file or filename']
    with pytest.raises((FileNotFoundError, AttributeError)):
        Estimator(analyzer=analyzer, input=input_type).fit_transform(data)