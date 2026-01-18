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
def test_format_cursor_data_BoundaryNorm():
    """Test if cursor data is correct when using BoundaryNorm."""
    X = np.empty((3, 3))
    X[0, 0] = 0.9
    X[0, 1] = 0.99
    X[0, 2] = 0.999
    X[1, 0] = -1
    X[1, 1] = 0
    X[1, 2] = 1
    X[2, 0] = 0.09
    X[2, 1] = 0.009
    X[2, 2] = 0.0009
    fig, ax = plt.subplots()
    fig.suptitle('-1..1 to 0..256 in 0.1')
    norm = mcolors.BoundaryNorm(np.linspace(-1, 1, 20), 256)
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    labels_list = ['[0.9]', '[1.]', '[1.]', '[-1.0]', '[0.0]', '[1.0]', '[0.09]', '[0.009]', '[0.0009]']
    for v, label in zip(X.flat, labels_list):
        assert img.format_cursor_data(v) == label
    plt.close()
    fig, ax = plt.subplots()
    fig.suptitle('-1..1 to 0..256 in 0.01')
    cmap = mpl.colormaps['RdBu_r'].resampled(200)
    norm = mcolors.BoundaryNorm(np.linspace(-1, 1, 200), 200)
    img = ax.imshow(X, cmap=cmap, norm=norm)
    labels_list = ['[0.90]', '[0.99]', '[1.0]', '[-1.00]', '[0.00]', '[1.00]', '[0.09]', '[0.009]', '[0.0009]']
    for v, label in zip(X.flat, labels_list):
        assert img.format_cursor_data(v) == label
    plt.close()
    fig, ax = plt.subplots()
    fig.suptitle('-1..1 to 0..256 in 0.001')
    cmap = mpl.colormaps['RdBu_r'].resampled(2000)
    norm = mcolors.BoundaryNorm(np.linspace(-1, 1, 2000), 2000)
    img = ax.imshow(X, cmap=cmap, norm=norm)
    labels_list = ['[0.900]', '[0.990]', '[0.999]', '[-1.000]', '[0.000]', '[1.000]', '[0.090]', '[0.009]', '[0.0009]']
    for v, label in zip(X.flat, labels_list):
        assert img.format_cursor_data(v) == label
    plt.close()
    X = np.empty((7, 1))
    X[0] = -1.0
    X[1] = 0.0
    X[2] = 0.1
    X[3] = 0.5
    X[4] = 0.9
    X[5] = 1.0
    X[6] = 2.0
    labels_list = ['[-1.0]', '[0.0]', '[0.1]', '[0.5]', '[0.9]', '[1.0]', '[2.0]']
    fig, ax = plt.subplots()
    fig.suptitle('noclip, neither')
    norm = mcolors.BoundaryNorm(np.linspace(0, 1, 4, endpoint=True), 256, clip=False, extend='neither')
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    for v, label in zip(X.flat, labels_list):
        assert img.format_cursor_data(v) == label
    plt.close()
    fig, ax = plt.subplots()
    fig.suptitle('noclip, min')
    norm = mcolors.BoundaryNorm(np.linspace(0, 1, 4, endpoint=True), 256, clip=False, extend='min')
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    for v, label in zip(X.flat, labels_list):
        assert img.format_cursor_data(v) == label
    plt.close()
    fig, ax = plt.subplots()
    fig.suptitle('noclip, max')
    norm = mcolors.BoundaryNorm(np.linspace(0, 1, 4, endpoint=True), 256, clip=False, extend='max')
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    for v, label in zip(X.flat, labels_list):
        assert img.format_cursor_data(v) == label
    plt.close()
    fig, ax = plt.subplots()
    fig.suptitle('noclip, both')
    norm = mcolors.BoundaryNorm(np.linspace(0, 1, 4, endpoint=True), 256, clip=False, extend='both')
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    for v, label in zip(X.flat, labels_list):
        assert img.format_cursor_data(v) == label
    plt.close()
    fig, ax = plt.subplots()
    fig.suptitle('clip, neither')
    norm = mcolors.BoundaryNorm(np.linspace(0, 1, 4, endpoint=True), 256, clip=True, extend='neither')
    img = ax.imshow(X, cmap='RdBu_r', norm=norm)
    for v, label in zip(X.flat, labels_list):
        assert img.format_cursor_data(v) == label
    plt.close()