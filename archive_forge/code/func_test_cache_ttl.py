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
@pytest.mark.xdist_group('cache')
@pytest.mark.parametrize('to_disk', (True, False))
def test_cache_ttl(to_disk):
    if to_disk and diskcache is None:
        pytest.skip('requires diskcache')
    global OFFSET
    OFFSET.clear()
    fn = cache(function_with_args, ttl=0.1, to_disk=to_disk)
    assert fn(0, 0) == 0
    time.sleep(0.2)
    assert fn(0, 0) == 1