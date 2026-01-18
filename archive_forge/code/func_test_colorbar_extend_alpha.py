import platform
import numpy as np
import pytest
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib as mpl
from matplotlib import rc_context
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
from matplotlib.colors import (
from matplotlib.colorbar import Colorbar
from matplotlib.ticker import FixedLocator, LogFormatter, StrMethodFormatter
from matplotlib.testing.decorators import check_figures_equal
@image_comparison(['colorbar_extend_alpha.png'], remove_text=True, savefig_kwarg={'dpi': 40})
def test_colorbar_extend_alpha():
    fig, ax = plt.subplots()
    im = ax.imshow([[0, 1], [2, 3]], alpha=0.3, interpolation='none')
    fig.colorbar(im, extend='both', boundaries=[0.5, 1.5, 2.5])