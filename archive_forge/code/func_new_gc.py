import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def new_gc(self):
    self.gc.ctx.save()
    self.gc._alpha = 1
    self.gc._forced_alpha = False
    return self.gc