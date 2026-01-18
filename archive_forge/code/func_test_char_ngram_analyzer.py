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
def test_char_ngram_analyzer():
    cnga = CountVectorizer(analyzer='char', strip_accents='unicode', ngram_range=(3, 6)).build_analyzer()
    text = "J'ai mangé du kangourou  ce midi, c'était pas très bon"
    expected = ["j'a", "'ai", 'ai ', 'i m', ' ma']
    assert cnga(text)[:5] == expected
    expected = ['s tres', ' tres ', 'tres b', 'res bo', 'es bon']
    assert cnga(text)[-5:] == expected
    text = 'This \n\tis a test, really.\n\n I met Harry yesterday'
    expected = ['thi', 'his', 'is ', 's i', ' is']
    assert cnga(text)[:5] == expected
    expected = [' yeste', 'yester', 'esterd', 'sterda', 'terday']
    assert cnga(text)[-5:] == expected
    cnga = CountVectorizer(input='file', analyzer='char', ngram_range=(3, 6)).build_analyzer()
    text = StringIO('This is a test with a file-like object!')
    expected = ['thi', 'his', 'is ', 's i', ' is']
    assert cnga(text)[:5] == expected