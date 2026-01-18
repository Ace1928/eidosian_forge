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
@pytest.mark.parametrize('fallback,fontlist', [('cm', ['DejaVu Sans', 'mpltest', 'STIXGeneral', 'cmr10', 'STIXGeneral']), ('stix', ['DejaVu Sans', 'mpltest', 'STIXGeneral'])])
def test_mathtext_fallback(fallback, fontlist):
    mpl.font_manager.fontManager.addfont(str(Path(__file__).resolve().parent / 'mpltest.ttf'))
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['mathtext.fontset'] = 'custom'
    mpl.rcParams['mathtext.rm'] = 'mpltest'
    mpl.rcParams['mathtext.it'] = 'mpltest:italic'
    mpl.rcParams['mathtext.bf'] = 'mpltest:bold'
    mpl.rcParams['mathtext.bfit'] = 'mpltest:italic:bold'
    mpl.rcParams['mathtext.fallback'] = fallback
    test_str = 'a$A\\AA\\breve\\gimel$'
    buff = io.BytesIO()
    fig, ax = plt.subplots()
    fig.text(0.5, 0.5, test_str, fontsize=40, ha='center')
    fig.savefig(buff, format='svg')
    tspans = ET.fromstring(buff.getvalue()).findall('.//{http://www.w3.org/2000/svg}tspan[@style]')
    char_fonts = [shlex.split(tspan.attrib['style'])[-1] for tspan in tspans]
    assert char_fonts == fontlist
    mpl.font_manager.fontManager.ttflist.pop()