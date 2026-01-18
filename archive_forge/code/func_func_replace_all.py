import re
import sys
import numpy as np
import pytest
from matplotlib import _preprocess_data
from matplotlib.axes import Axes
from matplotlib.testing import subprocess_run_for_testing
from matplotlib.testing.decorators import check_figures_equal
@_preprocess_data(label_namer='y')
def func_replace_all(ax, x, y, ls='x', label=None, w='NOT'):
    return f'x: {list(x)}, y: {list(y)}, ls: {ls}, w: {w}, label: {label}'