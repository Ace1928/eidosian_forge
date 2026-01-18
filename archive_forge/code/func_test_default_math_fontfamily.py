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
def test_default_math_fontfamily():
    mpl.rcParams['mathtext.fontset'] = 'cm'
    test_str = 'abc$abc\\alpha$'
    fig, ax = plt.subplots()
    text1 = fig.text(0.1, 0.1, test_str, font='Arial')
    prop1 = text1.get_fontproperties()
    assert prop1.get_math_fontfamily() == 'cm'
    text2 = fig.text(0.2, 0.2, test_str, fontproperties='Arial')
    prop2 = text2.get_fontproperties()
    assert prop2.get_math_fontfamily() == 'cm'
    fig.draw_without_rendering()