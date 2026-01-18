import datetime as dt
import io
import pathlib
import time
from collections import Counter
import numpy as np
import pandas as pd
import param
import pytest
import requests
from panel.io.cache import _find_hash_func, cache
from panel.io.state import set_curdoc, state
from panel.tests.util import serve_and_wait
def test_ndarray_hash():
    assert hashes_equal(np.array([0, 1, 2]), np.array([0, 1, 2]))
    assert not hashes_equal(np.array([0, 1, 2], dtype='uint32'), np.array([0, 1, 2], dtype='float64'))
    assert not hashes_equal(np.array([0, 1, 2]), np.array([2, 1, 0]))