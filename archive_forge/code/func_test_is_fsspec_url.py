import codecs
import errno
from functools import partial
from io import (
import mmap
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
import pytest
from pandas.compat import is_platform_windows
import pandas.util._test_decorators as td
import pandas as pd
import pandas._testing as tm
import pandas.io.common as icom
def test_is_fsspec_url():
    assert icom.is_fsspec_url('gcs://pandas/somethingelse.com')
    assert icom.is_fsspec_url('gs://pandas/somethingelse.com')
    assert not icom.is_fsspec_url('http://pandas/somethingelse.com')
    assert not icom.is_fsspec_url('random:pandas/somethingelse.com')
    assert not icom.is_fsspec_url('/local/path')
    assert not icom.is_fsspec_url('relative/local/path')
    assert not icom.is_fsspec_url('this is not fsspec://url')
    assert not icom.is_fsspec_url("{'url': 'gs://pandas/somethingelse.com'}")
    assert icom.is_fsspec_url('RFC-3986+compliant.spec://something')