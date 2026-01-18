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
@pytest.mark.parametrize('err, xycoords, match', ((TypeError, print, 'xycoords callable must return a BboxBase or Transform, not a'), (TypeError, [0, 0], "'xycoords' must be an instance of str, tuple"), (ValueError, 'foo', "'foo' is not a valid coordinate"), (ValueError, 'foo bar', "'foo bar' is not a valid coordinate"), (ValueError, 'offset foo', 'xycoords cannot be an offset coordinate'), (ValueError, 'axes foo', "'foo' is not a recognized unit")))
def test_annotate_errors(err, xycoords, match):
    fig, ax = plt.subplots()
    with pytest.raises(err, match=match):
        ax.annotate('xy', (0, 0), xytext=(0.5, 0.5), xycoords=xycoords)
        fig.canvas.draw()