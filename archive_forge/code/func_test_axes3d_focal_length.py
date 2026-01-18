import functools
import itertools
import platform
import pytest
from mpl_toolkits.mplot3d import Axes3D, axes3d, proj3d, art3d
import matplotlib as mpl
from matplotlib.backend_bases import (MouseButton, MouseEvent,
from matplotlib import cm
from matplotlib import colors as mcolors, patches as mpatch
from matplotlib.testing.decorators import image_comparison, check_figures_equal
from matplotlib.testing.widgets import mock_event
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path
from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np
@mpl3d_image_comparison(['axes3d_focal_length.png'], remove_text=False, style='mpl20')
def test_axes3d_focal_length():
    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'})
    axs[0].set_proj_type('persp', focal_length=np.inf)
    axs[1].set_proj_type('persp', focal_length=0.15)