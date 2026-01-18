import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_constrained_toggle():
    fig, ax = plt.subplots()
    with pytest.warns(PendingDeprecationWarning):
        fig.set_constrained_layout(True)
        assert fig.get_constrained_layout()
        fig.set_constrained_layout(False)
        assert not fig.get_constrained_layout()
        fig.set_constrained_layout(True)
        assert fig.get_constrained_layout()