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
@pytest.mark.parametrize('fontname, fontfile', [('DejaVu Sans', 'DejaVuSans.ttf'), ('WenQuanYi Zen Hei', 'wqy-zenhei.ttc')])
@pytest.mark.parametrize('fonttype', [3, 42])
def test_embed_fonts(fontname, fontfile, fonttype):
    if Path(findfont(FontProperties(family=[fontname]))).name != fontfile:
        pytest.skip(f'Font {fontname!r} may be missing')
    rcParams['pdf.fonttype'] = fonttype
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    ax.set_title('Axes Title', font=fontname)
    fig.savefig(io.BytesIO(), format='pdf')