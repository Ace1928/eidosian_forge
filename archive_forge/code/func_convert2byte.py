from __future__ import annotations
import os
import struct
import sys
from . import Image, ImageFile
def convert2byte(self, depth=255):
    minimum, maximum = self.getextrema()
    m = 1
    if maximum != minimum:
        m = depth / (maximum - minimum)
    b = -m * minimum
    return self.point(lambda i, m=m, b=b: i * m + b).convert('L')