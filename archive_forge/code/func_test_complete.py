from io import BytesIO
import ast
import pickle
import pickletools
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import cm
from matplotlib.testing import subprocess_run_helper
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.dates import rrulewrapper
from matplotlib.lines import VertexSelector
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.figure as mfigure
from mpl_toolkits.axes_grid1 import parasite_axes  # type: ignore
@mpl.style.context('default')
@check_figures_equal(extensions=['png'])
def test_complete(fig_test, fig_ref):
    _generate_complete_test_figure(fig_ref)
    pkl = pickle.dumps(fig_ref, pickle.HIGHEST_PROTOCOL)
    assert 'FigureCanvasAgg' not in [arg for op, arg, pos in pickletools.genops(pkl)]
    loaded = pickle.loads(pkl)
    loaded.canvas.draw()
    fig_test.set_size_inches(loaded.get_size_inches())
    fig_test.figimage(loaded.canvas.renderer.buffer_rgba())
    plt.close(loaded)