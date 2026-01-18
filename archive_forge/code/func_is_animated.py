from __future__ import annotations
import os
import re
from . import Image, ImageFile, ImagePalette
@property
def is_animated(self):
    return self.info[FRAMES] > 1