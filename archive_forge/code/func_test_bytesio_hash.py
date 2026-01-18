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
def test_bytesio_hash():
    bio1, bio2 = (io.BytesIO(), io.BytesIO())
    bio1.write(b'foo')
    bio2.write(b'foo')
    bio1.seek(0)
    bio2.seek(0)
    assert hashes_equal(bio1, bio2)
    bio3 = io.BytesIO()
    bio3.write(b'bar')
    bio3.seek(0)
    assert not hashes_equal(bio1, bio3)