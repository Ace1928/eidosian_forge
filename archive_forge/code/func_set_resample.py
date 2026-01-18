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
def set_resample(self, v):
    """
        Set whether image resampling is used.

        Parameters
        ----------
        v : bool or None
            If None, use :rc:`image.resample`.
        """
    v = mpl._val_or_rc(v, 'image.resample')
    self._resample = v
    self.stale = True