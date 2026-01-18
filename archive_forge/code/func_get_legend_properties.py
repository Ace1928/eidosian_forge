import itertools
import io
import base64
import numpy as np
import warnings
import matplotlib
from matplotlib.colors import colorConverter
from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
from matplotlib import ticker
def get_legend_properties(ax, legend):
    handles, labels = ax.get_legend_handles_labels()
    visible = legend.get_visible()
    return {'handles': handles, 'labels': labels, 'visible': visible}