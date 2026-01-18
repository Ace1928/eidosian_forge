import math
import os
import logging
from pathlib import Path
import warnings
import numpy as np
import PIL.Image
import PIL.PngImagePlugin
import matplotlib as mpl
from matplotlib import _api, cbook, cm
from matplotlib import _image
from matplotlib._image import *
import matplotlib.artist as martist
from matplotlib.backend_bases import FigureCanvasBase
import matplotlib.colors as mcolors
from matplotlib.transforms import (
def set_interpolation_stage(self, s):
    """
        Set when interpolation happens during the transform to RGBA.

        Parameters
        ----------
        s : {'data', 'rgba'} or None
            Whether to apply up/downsampling interpolation in data or RGBA
            space.
        """
    if s is None:
        s = 'data'
    _api.check_in_list(['data', 'rgba'], s=s)
    self._interpolation_stage = s
    self.stale = True