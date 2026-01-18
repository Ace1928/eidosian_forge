import functools
import logging
import math
import pathlib
import sys
import weakref
import numpy as np
import PIL.Image
import matplotlib as mpl
from matplotlib.backend_bases import (
from matplotlib import _api, cbook, backend_tools
from matplotlib._pylab_helpers import Gcf
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
import wx
def handle_clip_rectangle(self, gc):
    new_bounds = gc.get_clip_rectangle()
    if new_bounds is not None:
        new_bounds = new_bounds.bounds
    gfx_ctx = gc.gfx_ctx
    if gfx_ctx._lastcliprect != new_bounds:
        gfx_ctx._lastcliprect = new_bounds
        if new_bounds is None:
            gfx_ctx.ResetClip()
        else:
            gfx_ctx.Clip(new_bounds[0], self.height - new_bounds[1] - new_bounds[3], new_bounds[2], new_bounds[3])