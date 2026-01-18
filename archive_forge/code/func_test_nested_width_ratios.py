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
def test_nested_width_ratios(self):
    x = [['A', [['B'], ['C']]]]
    width_ratios = [2, 1]
    fig, axd = plt.subplot_mosaic(x, width_ratios=width_ratios)
    assert axd['A'].get_gridspec().get_width_ratios() == width_ratios
    assert axd['B'].get_gridspec().get_width_ratios() != width_ratios