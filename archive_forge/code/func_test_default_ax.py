import pathlib
import shutil
import string
from tempfile import mkdtemp
import numpy as np
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from cirq.devices import grid_qubit
from cirq.vis import heatmap
def test_default_ax():
    row_col_list = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    test_value_map = {grid_qubit.GridQubit(row, col): np.random.random() for row, col in row_col_list}
    _, _ = heatmap.Heatmap(test_value_map).plot()