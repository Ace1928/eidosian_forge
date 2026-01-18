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
@image_comparison(['colorbar_extensions_uniform.png', 'colorbar_extensions_proportional.png'], tol=1.0)
def test_colorbar_extension_length():
    """Test variable length colorbar extensions."""
    plt.rcParams['pcolormesh.snap'] = False
    _colorbar_extension_length('uniform')
    _colorbar_extension_length('proportional')