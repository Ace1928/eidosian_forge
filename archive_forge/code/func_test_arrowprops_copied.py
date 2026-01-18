from collections import namedtuple
import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.backend_bases import MouseButton, MouseEvent
from matplotlib.offsetbox import (
def test_arrowprops_copied():
    da = DrawingArea(20, 20, 0, 0, clip=True)
    arrowprops = {'arrowstyle': '->', 'relpos': (0.3, 0.7)}
    ab = AnnotationBbox(da, [0.5, 0.5], xybox=(-0.2, 0.5), xycoords='data', boxcoords='axes fraction', box_alignment=(0.0, 0.5), arrowprops=arrowprops)
    assert ab.arrowprops is not ab
    assert arrowprops['relpos'] == (0.3, 0.7)