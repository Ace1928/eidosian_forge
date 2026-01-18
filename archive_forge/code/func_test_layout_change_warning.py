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
@pytest.mark.parametrize('layout', ['constrained', 'compressed'])
def test_layout_change_warning(layout):
    """
    Raise a warning when a previously assigned layout changes to tight using
    plt.tight_layout().
    """
    fig, ax = plt.subplots(layout=layout)
    with pytest.warns(UserWarning, match='The figure layout has changed to'):
        plt.tight_layout()