import re
import numpy as np
import pytest
import pandas as pd
from pandas import DataFrame
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.util.version import Version
def test_if_scatterplot_colorbars_are_next_to_parent_axes(self):
    random_array = np.random.default_rng(2).random((10, 3))
    df = DataFrame(random_array, columns=['A label', 'B label', 'C label'])
    fig, axes = plt.subplots(1, 2)
    df.plot.scatter('A label', 'B label', c='C label', ax=axes[0])
    df.plot.scatter('A label', 'B label', c='C label', ax=axes[1])
    plt.tight_layout()
    points = np.array([ax.get_position().get_points() for ax in fig.axes])
    axes_x_coords = points[:, :, 0]
    parent_distance = axes_x_coords[1, :] - axes_x_coords[0, :]
    colorbar_distance = axes_x_coords[3, :] - axes_x_coords[2, :]
    assert np.isclose(parent_distance, colorbar_distance, atol=1e-07).all()