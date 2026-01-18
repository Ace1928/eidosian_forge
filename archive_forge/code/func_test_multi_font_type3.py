import datetime
import decimal
import io
import os
from pathlib import Path
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import (
from matplotlib.cbook import _get_data_path
from matplotlib.ft2font import FT2Font
from matplotlib.font_manager import findfont, FontProperties
from matplotlib.backends._backend_pdf_ps import get_glyphs_subset
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
@image_comparison(['multi_font_type3.pdf'], tol=4.6)
def test_multi_font_type3():
    fp = fm.FontProperties(family=['WenQuanYi Zen Hei'])
    if Path(fm.findfont(fp)).name != 'wqy-zenhei.ttc':
        pytest.skip('Font may be missing')
    plt.rc('font', family=['DejaVu Sans', 'WenQuanYi Zen Hei'], size=27)
    plt.rc('pdf', fonttype=3)
    fig = plt.figure()
    fig.text(0.15, 0.475, 'There are 几个汉字 in between!')