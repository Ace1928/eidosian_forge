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
def test_multipage_properfinalize():
    pdfio = io.BytesIO()
    with PdfPages(pdfio) as pdf:
        for i in range(10):
            fig, ax = plt.subplots()
            ax.set_title('This is a long title')
            fig.savefig(pdf, format='pdf')
    s = pdfio.getvalue()
    assert s.count(b'startxref') == 1
    assert len(s) < 40000