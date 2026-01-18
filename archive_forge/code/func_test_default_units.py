import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
def test_default_units(self):
    assert isinstance(self.cc.default_units(['a'], self.ax), cat.UnitData)