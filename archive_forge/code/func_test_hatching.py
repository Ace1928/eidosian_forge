import io
from itertools import chain
import numpy as np
import pytest
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath
import matplotlib.transforms as mtransforms
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.backend_bases as mbackend_bases
import matplotlib as mpl
from matplotlib.testing.decorators import check_figures_equal, image_comparison
@image_comparison(['hatching'], remove_text=True, style='default')
def test_hatching():
    fig, ax = plt.subplots(1, 1)
    rect1 = mpatches.Rectangle((0, 0), 3, 4, hatch='/')
    ax.add_patch(rect1)
    rect2 = mcollections.RegularPolyCollection(4, sizes=[16000], offsets=[(1.5, 6.5)], offset_transform=ax.transData, hatch='/')
    ax.add_collection(rect2)
    rect3 = mpatches.Rectangle((4, 0), 3, 4, hatch='/', edgecolor='C1')
    ax.add_patch(rect3)
    rect4 = mcollections.RegularPolyCollection(4, sizes=[16000], offsets=[(5.5, 6.5)], offset_transform=ax.transData, hatch='/', edgecolor='C1')
    ax.add_collection(rect4)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 9)