import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_discouraged_api():
    fig, ax = plt.subplots(constrained_layout=True)
    fig.draw_without_rendering()
    with pytest.warns(PendingDeprecationWarning, match='will be deprecated'):
        fig, ax = plt.subplots()
        fig.set_constrained_layout(True)
        fig.draw_without_rendering()
    with pytest.warns(PendingDeprecationWarning, match='will be deprecated'):
        fig, ax = plt.subplots()
        fig.set_constrained_layout({'w_pad': 0.02, 'h_pad': 0.02})
        fig.draw_without_rendering()