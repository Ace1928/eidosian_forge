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
def test_get_constrained_layout_pads():
    params = {'w_pad': 0.01, 'h_pad': 0.02, 'wspace': 0.03, 'hspace': 0.04}
    expected = tuple([*params.values()])
    fig = plt.figure(layout=mpl.layout_engine.ConstrainedLayoutEngine(**params))
    with pytest.warns(PendingDeprecationWarning, match='will be deprecated'):
        assert fig.get_constrained_layout_pads() == expected