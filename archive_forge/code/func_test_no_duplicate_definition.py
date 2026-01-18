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
def test_no_duplicate_definition():
    fig = Figure()
    axs = fig.subplots(4, 4, subplot_kw=dict(projection='polar'))
    for ax in axs.flat:
        ax.set(xticks=[], yticks=[])
        ax.plot([1, 2])
    fig.suptitle('hello, world')
    buf = io.StringIO()
    fig.savefig(buf, format='eps')
    buf.seek(0)
    wds = [ln.partition(' ')[0] for ln in buf.readlines() if ln.startswith('/')]
    assert max(Counter(wds).values()) == 1