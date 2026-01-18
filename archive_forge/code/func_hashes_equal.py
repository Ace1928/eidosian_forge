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
def hashes_equal(v1, v2):
    a, b = (_find_hash_func(v1)(v1), _find_hash_func(v2)(v2))
    return a == b