import pytest
from numpy.testing import assert_allclose, assert_array_equal
from matplotlib.sankey import Sankey
from matplotlib.testing.decorators import check_figures_equal
def show_three_decimal_places(value):
    return f'{value:.3f}'