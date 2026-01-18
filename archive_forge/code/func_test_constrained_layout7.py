import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_constrained_layout7():
    """Test for proper warning if fig not set in GridSpec"""
    with pytest.warns(UserWarning, match='There are no gridspecs with layoutgrids. Possibly did not call parent GridSpec with the "figure" keyword'):
        fig = plt.figure(layout='constrained')
        gs = gridspec.GridSpec(1, 2)
        gsl = gridspec.GridSpecFromSubplotSpec(2, 2, gs[0])
        gsr = gridspec.GridSpecFromSubplotSpec(1, 2, gs[1])
        for gs in gsl:
            fig.add_subplot(gs)
        fig.draw_without_rendering()