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
def test_fignum_exists():
    plt.figure('one')
    plt.figure(2)
    plt.figure('three')
    plt.figure()
    assert plt.fignum_exists('one')
    assert plt.fignum_exists(2)
    assert plt.fignum_exists('three')
    assert plt.fignum_exists(4)
    plt.close('one')
    plt.close(4)
    assert not plt.fignum_exists('one')
    assert not plt.fignum_exists(4)