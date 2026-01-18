import functools
import gzip
import math
import numpy as np
from .. import _api, cbook, font_manager
from matplotlib.backend_bases import (
from matplotlib.font_manager import ttfFontProperty
from matplotlib.path import Path
from matplotlib.transforms import Affine2D
def get_text_width_height_descent(self, s, prop, ismath):
    if ismath == 'TeX':
        return super().get_text_width_height_descent(s, prop, ismath)
    if ismath:
        width, height, descent, *_ = self._text2path.mathtext_parser.parse(s, self.dpi, prop)
        return (width, height, descent)
    ctx = self.text_ctx
    ctx.save()
    ctx.select_font_face(*_cairo_font_args_from_font_prop(prop))
    ctx.set_font_size(self.points_to_pixels(prop.get_size_in_points()))
    y_bearing, w, h = ctx.text_extents(s)[1:4]
    ctx.restore()
    return (w, h, h + y_bearing)