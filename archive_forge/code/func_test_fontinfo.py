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
def test_fontinfo():
    fontpath = mpl.font_manager.findfont('DejaVu Sans')
    font = mpl.ft2font.FT2Font(fontpath)
    table = font.get_sfnt_table('head')
    assert table is not None
    assert table['version'] == (1, 0)