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
def test_clf_keyword():
    text1 = 'A fancy plot'
    text2 = 'Really fancy!'
    fig0 = plt.figure(num=1)
    fig0.suptitle(text1)
    assert [t.get_text() for t in fig0.texts] == [text1]
    fig1 = plt.figure(num=1, clear=False)
    fig1.text(0.5, 0.5, text2)
    assert fig0 is fig1
    assert [t.get_text() for t in fig1.texts] == [text1, text2]
    fig2, ax2 = plt.subplots(2, 1, num=1, clear=True)
    assert fig0 is fig2
    assert [t.get_text() for t in fig2.texts] == []