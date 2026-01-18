import sys
from types import SimpleNamespace
from unittest.mock import MagicMock
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.backends import backend_template
from matplotlib.backends.backend_template import (
def test_load_template():
    mpl.use('template')
    assert type(plt.figure().canvas) == FigureCanvasTemplate