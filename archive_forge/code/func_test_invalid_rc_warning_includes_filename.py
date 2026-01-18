from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt, style
from matplotlib.style.core import USER_LIBRARY_PATHS, STYLE_EXTENSION
def test_invalid_rc_warning_includes_filename(caplog):
    SETTINGS = {'foo': 'bar'}
    basename = 'basename'
    with temp_style(basename, SETTINGS):
        pass
    assert len(caplog.records) == 1 and basename in caplog.records[0].getMessage()