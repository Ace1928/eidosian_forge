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
def test_gridspec_no_mutate_input():
    gs = {'left': 0.1}
    gs_orig = dict(gs)
    plt.subplots(1, 2, width_ratios=[1, 2], gridspec_kw=gs)
    assert gs == gs_orig
    plt.subplot_mosaic('AB', width_ratios=[1, 2], gridspec_kw=gs)