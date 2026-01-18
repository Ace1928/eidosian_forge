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
def get_figure_properties(fig):
    return {'figwidth': fig.get_figwidth(), 'figheight': fig.get_figheight(), 'dpi': fig.dpi}