import copy
import matplotlib.pyplot as plt
from matplotlib.scale import (
import matplotlib.scale as mscale
from matplotlib.ticker import AsinhLocator, LogFormatterSciNotation
from matplotlib.testing.decorators import check_figures_equal, image_comparison
import numpy as np
from numpy.testing import assert_allclose
import io
import pytest
def test_extra_kwargs_raise():
    fig, ax = plt.subplots()
    for scale in ['linear', 'log', 'symlog']:
        with pytest.raises(TypeError):
            ax.set_yscale(scale, foo='mask')