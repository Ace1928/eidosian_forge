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
def test_base_init(self):
    fig, ax = plt.subplots()
    s3 = AsinhScale(axis=None, base=3)
    assert s3._base == 3
    assert s3._subs == (2,)
    s7 = AsinhScale(axis=None, base=7, subs=(2, 4))
    assert s7._base == 7
    assert s7._subs == (2, 4)