from datetime import datetime
import io
import warnings
import numpy as np
from numpy.testing import assert_almost_equal
from packaging.version import parse as parse_version
import pyparsing
import pytest
import matplotlib as mpl
from matplotlib.backend_bases import MouseEvent
from matplotlib.font_manager import FontProperties
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
from matplotlib.text import Text, Annotation, OffsetFrom
@image_comparison(['font_styles'])
def test_font_styles():

    def find_matplotlib_font(**kw):
        prop = FontProperties(**kw)
        path = findfont(prop, directory=mpl.get_data_path())
        return FontProperties(fname=path)
    from matplotlib.font_manager import FontProperties, findfont
    warnings.filterwarnings('ignore', "findfont: Font family \\[u?'Foo'\\] not found. Falling back to .", UserWarning, module='matplotlib.font_manager')
    fig, ax = plt.subplots()
    normal_font = find_matplotlib_font(family='sans-serif', style='normal', variant='normal', size=14)
    a = ax.annotate('Normal Font', (0.1, 0.1), xycoords='axes fraction', fontproperties=normal_font)
    assert a.get_fontname() == 'DejaVu Sans'
    assert a.get_fontstyle() == 'normal'
    assert a.get_fontvariant() == 'normal'
    assert a.get_weight() == 'normal'
    assert a.get_stretch() == 'normal'
    bold_font = find_matplotlib_font(family='Foo', style='normal', variant='normal', weight='bold', stretch=500, size=14)
    ax.annotate('Bold Font', (0.1, 0.2), xycoords='axes fraction', fontproperties=bold_font)
    bold_italic_font = find_matplotlib_font(family='sans serif', style='italic', variant='normal', weight=750, stretch=500, size=14)
    ax.annotate('Bold Italic Font', (0.1, 0.3), xycoords='axes fraction', fontproperties=bold_italic_font)
    light_font = find_matplotlib_font(family='sans-serif', style='normal', variant='normal', weight=200, stretch=500, size=14)
    ax.annotate('Light Font', (0.1, 0.4), xycoords='axes fraction', fontproperties=light_font)
    condensed_font = find_matplotlib_font(family='sans-serif', style='normal', variant='normal', weight=500, stretch=100, size=14)
    ax.annotate('Condensed Font', (0.1, 0.5), xycoords='axes fraction', fontproperties=condensed_font)
    ax.set_xticks([])
    ax.set_yticks([])