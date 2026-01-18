import gzip
import os
import shutil
from bz2 import BZ2File
from importlib import resources
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
import sklearn
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_svmlight_files
from sklearn.utils._testing import (
from sklearn.utils.fixes import CSR_CONTAINERS
def test_dump_comment():
    X, y = _load_svmlight_local_test_file(datafile)
    X = X.toarray()
    f = BytesIO()
    ascii_comment = 'This is a comment\nspanning multiple lines.'
    dump_svmlight_file(X, y, f, comment=ascii_comment, zero_based=False)
    f.seek(0)
    X2, y2 = load_svmlight_file(f, zero_based=False)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)
    utf8_comment = b'It is true that\n\xc2\xbd\xc2\xb2 = \xc2\xbc'
    f = BytesIO()
    with pytest.raises(UnicodeDecodeError):
        dump_svmlight_file(X, y, f, comment=utf8_comment)
    unicode_comment = utf8_comment.decode('utf-8')
    f = BytesIO()
    dump_svmlight_file(X, y, f, comment=unicode_comment, zero_based=False)
    f.seek(0)
    X2, y2 = load_svmlight_file(f, zero_based=False)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)
    f = BytesIO()
    with pytest.raises(ValueError):
        dump_svmlight_file(X, y, f, comment="I've got a \x00.")