from __future__ import annotations
from packaging.version import Version
import numpy as np
from toolz import memoize
from datashader.glyphs.glyph import Glyph
from datashader.utils import isreal, ngjit
from numba import cuda
def required_columns(self):
    return [self.x, self.y]