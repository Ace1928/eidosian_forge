from tempfile import TemporaryFile
import numpy as np
from packaging.version import parse as parse_version
import pytest
import matplotlib as mpl
from matplotlib import dviread
from matplotlib.testing import _has_tex_package
from matplotlib.testing.decorators import check_figures_equal, image_comparison
from matplotlib.testing._markers import needs_usetex
import matplotlib.pyplot as plt
@pytest.mark.parametrize('preamble', ['\\usepackage[full]{textcomp}', '\\usepackage{underscore}'])
def test_latex_pkg_already_loaded(preamble):
    plt.rcParams['text.latex.preamble'] = preamble
    fig = plt.figure()
    fig.text(0.5, 0.5, 'hello, world', usetex=True)
    fig.canvas.draw()