from collections import defaultdict
import functools
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.axes as maxes
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d
def calc_arrows(UVW):
    x = UVW[:, 0]
    y = UVW[:, 1]
    norm = np.linalg.norm(UVW[:, :2], axis=1)
    x_p = np.divide(y, norm, where=norm != 0, out=np.zeros_like(x))
    y_p = np.divide(-x, norm, where=norm != 0, out=np.ones_like(x))
    rangle = math.radians(15)
    c = math.cos(rangle)
    s = math.sin(rangle)
    r13 = y_p * s
    r32 = x_p * s
    r12 = x_p * y_p * (1 - c)
    Rpos = np.array([[c + x_p ** 2 * (1 - c), r12, r13], [r12, c + y_p ** 2 * (1 - c), -r32], [-r13, r32, np.full_like(x_p, c)]])
    Rneg = Rpos.copy()
    Rneg[[0, 1, 2, 2], [2, 2, 0, 1]] *= -1
    Rpos_vecs = np.einsum('ij...,...j->...i', Rpos, UVW)
    Rneg_vecs = np.einsum('ij...,...j->...i', Rneg, UVW)
    return np.stack([Rpos_vecs, Rneg_vecs], axis=1)