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
@check_figures_equal(extensions=['svg', 'pdf', 'eps', 'png'])
def test_animated_with_canvas_change(fig_test, fig_ref):
    ax_ref = fig_ref.subplots()
    ax_ref.plot(range(5))
    ax_test = fig_test.subplots()
    ax_test.plot(range(5), animated=True)