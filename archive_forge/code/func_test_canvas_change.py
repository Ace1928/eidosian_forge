import re
from matplotlib import path, transforms
from matplotlib.backend_bases import (
from matplotlib.backend_tools import RubberbandBase
from matplotlib.figure import Figure
from matplotlib.testing._markers import needs_pgf_xelatex
import matplotlib.pyplot as plt
import numpy as np
import pytest
def test_canvas_change():
    fig = plt.figure()
    canvas = FigureCanvasBase(fig)
    plt.close(fig)
    assert not plt.fignum_exists(fig.number)