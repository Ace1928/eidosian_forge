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
def test_nested_height_ratios(self):
    x = [['A', [['B'], ['C']]], ['D', 'D']]
    height_ratios = [1, 2]
    fig, axd = plt.subplot_mosaic(x, height_ratios=height_ratios)
    assert axd['D'].get_gridspec().get_height_ratios() == height_ratios
    assert axd['B'].get_gridspec().get_height_ratios() != height_ratios