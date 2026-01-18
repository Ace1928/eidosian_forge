from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
def test_context_with_union_of_dict_and_namedstyle():
    original_value = 'gray'
    other_param = 'text.usetex'
    other_value = True
    d = {other_param: other_value}
    mpl.rcParams[PARAM] = original_value
    mpl.rcParams[other_param] = not other_value
    with temp_style('test', DUMMY_SETTINGS):
        with style.context(['test', d]):
            assert mpl.rcParams[PARAM] == VALUE
            assert mpl.rcParams[other_param] == other_value
    assert mpl.rcParams[PARAM] == original_value
    assert mpl.rcParams[other_param] == (not other_value)