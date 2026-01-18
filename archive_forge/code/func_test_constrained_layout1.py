import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
@image_comparison(['constrained_layout1.png'])
def test_constrained_layout1():
    """Test constrained_layout for a single subplot"""
    fig = plt.figure(layout='constrained')
    ax = fig.add_subplot()
    example_plot(ax, fontsize=24)