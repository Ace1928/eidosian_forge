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
@check_figures_equal(extensions=['png'])
def test_genfrac_displaystyle(fig_test, fig_ref):
    fig_test.text(0.1, 0.1, '$\\dfrac{2x}{3y}$')
    thickness = _mathtext.TruetypeFonts.get_underline_thickness(None, None, fontsize=mpl.rcParams['font.size'], dpi=mpl.rcParams['savefig.dpi'])
    fig_ref.text(0.1, 0.1, '$\\genfrac{}{}{%f}{0}{2x}{3y}$' % thickness)