import numpy as np
from numpy.testing import (
import numpy.ma.testutils as matest
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.path import Path
from matplotlib.testing.decorators import image_comparison, check_figures_equal
def gradient_quad(x, y):
    return (2 * a * (x - 0.5) + c * y, 2 * b * (y - 0.5) + c * x)