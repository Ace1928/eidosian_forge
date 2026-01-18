from __future__ import annotations
import re
from . import Image, _imagingmorph
def set_lut(self, lut):
    """Set the lut from an external source"""
    self.lut = lut