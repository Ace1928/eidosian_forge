from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
def test_context_with_dict_before_namedstyle():
    original_value = 'gray'
    other_value = 'blue'
    mpl.rcParams[PARAM] = original_value
    with temp_style('test', DUMMY_SETTINGS):
        with style.context([{PARAM: other_value}, 'test']):
            assert mpl.rcParams[PARAM] == VALUE
    assert mpl.rcParams[PARAM] == original_value