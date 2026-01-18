import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal
import pytest
import matplotlib as mpl
from matplotlib.patches import (Annulus, Ellipse, Patch, Polygon, Rectangle,
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from matplotlib import (
import sys
@image_comparison(['clip_to_bbox'])
def test_clip_to_bbox():
    fig, ax = plt.subplots()
    ax.set_xlim([-18, 20])
    ax.set_ylim([-150, 100])
    path = mpath.Path.unit_regular_star(8).deepcopy()
    path.vertices *= [10, 100]
    path.vertices -= [5, 25]
    path2 = mpath.Path.unit_circle().deepcopy()
    path2.vertices *= [10, 100]
    path2.vertices += [10, -25]
    combined = mpath.Path.make_compound_path(path, path2)
    patch = mpatches.PathPatch(combined, alpha=0.5, facecolor='coral', edgecolor='none')
    ax.add_patch(patch)
    bbox = mtransforms.Bbox([[-12, -77.5], [50, -110]])
    result_path = combined.clip_to_bbox(bbox)
    result_patch = mpatches.PathPatch(result_path, alpha=0.5, facecolor='green', lw=4, edgecolor='black')
    ax.add_patch(result_patch)