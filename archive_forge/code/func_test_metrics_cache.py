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
@needs_usetex
def test_metrics_cache():
    mpl.text._get_text_metrics_with_cache_impl.cache_clear()
    fig = plt.figure()
    fig.text(0.3, 0.5, 'foo\nbar')
    fig.text(0.3, 0.5, 'foo\nbar', usetex=True)
    fig.text(0.5, 0.5, 'foo\nbar', usetex=True)
    fig.canvas.draw()
    renderer = fig._get_renderer()
    ys = {}

    def call(*args, **kwargs):
        renderer, x, y, s, *_ = args
        ys.setdefault(s, set()).add(y)
    renderer.draw_tex = call
    fig.canvas.draw()
    assert [*ys] == ['foo', 'bar']
    assert len(ys['foo']) == len(ys['bar']) == 1
    info = mpl.text._get_text_metrics_with_cache_impl.cache_info()
    assert info.hits > info.misses