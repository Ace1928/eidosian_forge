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
@check_figures_equal(extensions=['png'])
def test_all_nested(self, fig_test, fig_ref):
    x = [['A', 'B'], ['C', 'D']]
    y = [['E', 'F'], ['G', 'H']]
    fig_ref.set_layout_engine('constrained')
    fig_test.set_layout_engine('constrained')
    grid_axes = fig_test.subplot_mosaic([[x, y]])
    for ax in grid_axes.values():
        ax.set_title(ax.get_label())
    gs = fig_ref.add_gridspec(1, 2)
    gs_left = gs[0, 0].subgridspec(2, 2)
    for j, r in enumerate(x):
        for k, label in enumerate(r):
            fig_ref.add_subplot(gs_left[j, k]).set_title(label)
    gs_right = gs[0, 1].subgridspec(2, 2)
    for j, r in enumerate(y):
        for k, label in enumerate(r):
            fig_ref.add_subplot(gs_right[j, k]).set_title(label)