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
def test_glyphs_subset():
    fpath = str(_get_data_path('fonts/ttf/DejaVuSerif.ttf'))
    chars = 'these should be subsetted! 1234567890'
    nosubfont = FT2Font(fpath)
    nosubfont.set_text(chars)
    subfont = FT2Font(get_glyphs_subset(fpath, chars))
    subfont.set_text(chars)
    nosubcmap = nosubfont.get_charmap()
    subcmap = subfont.get_charmap()
    assert {*chars} == {chr(key) for key in subcmap}
    assert len(subcmap) < len(nosubcmap)
    assert subfont.get_num_glyphs() == nosubfont.get_num_glyphs()