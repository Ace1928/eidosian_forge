from collections.abc import Iterable, Sequence
from contextlib import ExitStack
import functools
import inspect
import logging
from numbers import Real
from operator import attrgetter
import types
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, offsetbox
import matplotlib.artist as martist
import matplotlib.axis as maxis
from matplotlib.cbook import _OrderedSet, _check_1d, index_of
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
from matplotlib.gridspec import SubplotSpec
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.rcsetup import cycler, validate_axisbelow
import matplotlib.spines as mspines
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
def format_deltas(key, dx, dy):
    if key == 'control':
        if abs(dx) > abs(dy):
            dy = dx
        else:
            dx = dy
    elif key == 'x':
        dy = 0
    elif key == 'y':
        dx = 0
    elif key == 'shift':
        if 2 * abs(dx) < abs(dy):
            dx = 0
        elif 2 * abs(dy) < abs(dx):
            dy = 0
        elif abs(dx) > abs(dy):
            dy = dy / abs(dy) * abs(dx)
        else:
            dx = dx / abs(dx) * abs(dy)
    return (dx, dy)