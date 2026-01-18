import contextlib
import logging
import os
from pathlib import Path
import sys
import warnings
import matplotlib as mpl
from matplotlib import _api, _docstring, _rc_params_in_file, rcParamsDefault
def read_style_directory(style_dir):
    """Return dictionary of styles defined in *style_dir*."""
    styles = dict()
    for path in Path(style_dir).glob(f'*.{STYLE_EXTENSION}'):
        with warnings.catch_warnings(record=True) as warns:
            styles[path.stem] = _rc_params_in_file(path)
        for w in warns:
            _log.warning('In %s: %s', path, w.message)
    return styles