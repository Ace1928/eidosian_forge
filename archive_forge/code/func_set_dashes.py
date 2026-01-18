import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def set_dashes(self, offset, dashes):
    self._dashes = (offset, dashes)
    if dashes is None:
        self.ctx.set_dash([], 0)
    else:
        self.ctx.set_dash(list(self.renderer.points_to_pixels(np.asarray(dashes))), offset)