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
def test_inset_and_secondary():
    fig, ax = plt.subplots()
    ax.inset_axes([0.1, 0.1, 0.3, 0.3])
    ax.secondary_xaxis('top', functions=(np.square, np.sqrt))
    pickle.loads(pickle.dumps(fig))