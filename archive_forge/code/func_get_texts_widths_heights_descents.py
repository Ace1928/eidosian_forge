from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
def get_texts_widths_heights_descents(self, renderer):
    """
        Return a list of ``(width, height, descent)`` tuples for ticklabels.

        Empty labels are left out.
        """
    whd_list = []
    for _loc, _angle, label in self._locs_angles_labels:
        if not label.strip():
            continue
        clean_line, ismath = self._preprocess_math(label)
        whd = renderer.get_text_width_height_descent(clean_line, self._fontproperties, ismath=ismath)
        whd_list.append(whd)
    return whd_list