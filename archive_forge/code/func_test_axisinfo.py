import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
def test_axisinfo(self):
    axis = self.cc.axisinfo(self.unit, self.ax)
    assert isinstance(axis.majloc, cat.StrCategoryLocator)
    assert isinstance(axis.majfmt, cat.StrCategoryFormatter)