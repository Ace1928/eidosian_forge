from collections import Counter
from pathlib import Path
import io
import re
import tempfile
import numpy as np
import pytest
from matplotlib import cbook, path, patheffects, font_manager as fm
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from matplotlib.testing._markers import needs_ghostscript, needs_usetex
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib as mpl
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
@pytest.mark.parametrize('fonttype', ['3', '42'])
def test_fonttype(fonttype):
    mpl.rcParams['ps.fonttype'] = fonttype
    fig, ax = plt.subplots()
    ax.text(0.25, 0.5, 'Forty-two is the answer to everything!')
    buf = io.BytesIO()
    fig.savefig(buf, format='ps')
    test = b'/FontType ' + bytes(f'{fonttype}', encoding='utf-8') + b' def'
    assert re.search(test, buf.getvalue(), re.MULTILINE)