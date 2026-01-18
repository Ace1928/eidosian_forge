import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.parametrize('fvals', fvalues, ids=fids)
def test_convert_fail(self, fvals):
    with pytest.raises(TypeError):
        self.cc.convert(fvals, self.unit, self.ax)