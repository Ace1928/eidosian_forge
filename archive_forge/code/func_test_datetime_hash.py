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
def test_datetime_hash():
    assert hashes_equal(dt.datetime(1988, 4, 14, 12, 3, 2, 1), dt.datetime(1988, 4, 14, 12, 3, 2, 1))
    assert not hashes_equal(dt.datetime(1988, 4, 14, 12, 3, 2, 1), dt.datetime(1988, 4, 14, 12, 3, 2, 2))