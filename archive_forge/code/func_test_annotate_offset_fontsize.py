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
def test_annotate_offset_fontsize():
    fig, ax = plt.subplots()
    text_coords = ['offset points', 'offset fontsize']
    xy_text = [(10, 10), (1, 1)]
    anns = [ax.annotate('test', xy=(0.5, 0.5), xytext=xy_text[i], fontsize='10', xycoords='data', textcoords=text_coords[i]) for i in range(2)]
    points_coords, fontsize_coords = [ann.get_window_extent() for ann in anns]
    fig.canvas.draw()
    assert str(points_coords) == str(fontsize_coords)