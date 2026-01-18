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
@_api.rename_parameter('3.8', 's', 'filternorm')
def set_filternorm(self, filternorm):
    pass