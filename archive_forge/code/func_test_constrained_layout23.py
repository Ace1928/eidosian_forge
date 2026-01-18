import gc
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from matplotlib import gridspec, ticker
def test_constrained_layout23():
    """
    Comment in #11035: suptitle used to cause an exception when
    reusing a figure w/ CL with ``clear=True``.
    """
    for i in range(2):
        fig = plt.figure(layout='constrained', clear=True, num='123')
        gs = fig.add_gridspec(1, 2)
        sub = gs[0].subgridspec(2, 2)
        fig.suptitle(f'Suptitle{i}')