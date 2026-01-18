import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('value', ['hi', 'мир'], ids=['ascii', 'unicode'])
def test_convert_one_string(self, value):
    assert self.cc.convert(value, self.unit, self.ax) == 0