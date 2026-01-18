from __future__ import annotations
import io
from pathlib import Path
import platform
import re
import shlex
from xml.etree import ElementTree as ET
from typing import Any
import numpy as np
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
from matplotlib import mathtext, _mathtext
def test_mathtext_cmr10_minus_sign():
    mpl.rcParams['font.family'] = 'cmr10'
    mpl.rcParams['axes.formatter.use_mathtext'] = True
    fig, ax = plt.subplots()
    ax.plot(range(-1, 1), range(-1, 1))
    fig.canvas.draw()