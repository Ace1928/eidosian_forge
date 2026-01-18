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
def test_axes_removal():
    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[1].remove()
    axs[0].plot([datetime(2000, 1, 1), datetime(2000, 2, 1)], [0, 1])
    assert isinstance(axs[0].xaxis.get_major_formatter(), mdates.AutoDateFormatter)
    fig, axs = plt.subplots(1, 2, sharex=True)
    axs[1].xaxis.set_major_formatter(ScalarFormatter())
    axs[1].remove()
    axs[0].plot([datetime(2000, 1, 1), datetime(2000, 2, 1)], [0, 1])
    assert isinstance(axs[0].xaxis.get_major_formatter(), ScalarFormatter)