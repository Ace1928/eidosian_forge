import base64
import io
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest
from matplotlib.testing.decorators import (
import matplotlib.pyplot as plt
from matplotlib import patches, transforms
from matplotlib.path import Path
@check_figures_equal()
def test_closed_path_clipping(fig_test, fig_ref):
    vertices = []
    for roll in range(8):
        offset = 0.1 * roll + 0.1
        pattern = [[-0.5, 1.5], [-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [1 - offset / 2, 1.5], [1 - offset / 2, offset], [offset / 2, offset], [offset / 2, 1.5]]
        pattern = np.roll(pattern, roll, axis=0)
        pattern = np.concatenate((pattern, pattern[:1, :]))
        vertices.append(pattern)
    codes = np.full(len(vertices[0]), Path.LINETO)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    codes = np.tile(codes, len(vertices))
    vertices = np.concatenate(vertices)
    fig_test.set_size_inches((5, 5))
    path = Path(vertices, codes)
    fig_test.add_artist(patches.PathPatch(path, facecolor='none'))
    fig_ref.set_size_inches((5, 5))
    codes = codes.copy()
    codes[codes == Path.CLOSEPOLY] = Path.LINETO
    path = Path(vertices, codes)
    fig_ref.add_artist(patches.PathPatch(path, facecolor='none'))