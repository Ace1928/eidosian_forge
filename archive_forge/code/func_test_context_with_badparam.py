from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
def test_context_with_badparam():
    original_value = 'gray'
    other_value = 'blue'
    with style.context({PARAM: other_value}):
        assert mpl.rcParams[PARAM] == other_value
        x = style.context({PARAM: original_value, 'badparam': None})
        with pytest.raises(KeyError):
            with x:
                pass
        assert mpl.rcParams[PARAM] == other_value