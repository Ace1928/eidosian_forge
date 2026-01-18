import copy
from datetime import datetime
import io
from pathlib import Path
import pickle
import platform
from threading import Timer
from types import SimpleNamespace
import warnings
import numpy as np
import pytest
from PIL import Image
import matplotlib as mpl
from matplotlib import gridspec
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.axes import Axes
from matplotlib.backend_bases import KeyEvent, MouseEvent
from matplotlib.figure import Figure, FigureBase
from matplotlib.layout_engine import (ConstrainedLayoutEngine,
from matplotlib.ticker import AutoMinorLocator, FixedFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
def test_invalid_figure_add_axes():
    fig = plt.figure()
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'rect'"):
        fig.add_axes()
    with pytest.raises(ValueError):
        fig.add_axes((0.1, 0.1, 0.5, np.nan))
    with pytest.raises(TypeError, match="multiple values for argument 'rect'"):
        fig.add_axes([0, 0, 1, 1], rect=[0, 0, 1, 1])
    fig2, ax = plt.subplots()
    with pytest.raises(ValueError, match='The Axes must have been created in the present figure'):
        fig.add_axes(ax)
    fig2.delaxes(ax)
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='Passing more than one positional argument'):
        fig2.add_axes(ax, 'extra positional argument')
    with pytest.warns(mpl.MatplotlibDeprecationWarning, match='Passing more than one positional argument'):
        fig.add_axes([0, 0, 1, 1], 'extra positional argument')