from struct import pack, unpack, calcsize, error, Struct
import os
import sys
import time
import array
import tempfile
import logging
import io
from datetime import date
import zipfile
def pointm(self, x, y, m=None):
    """Creates a POINTM shape.
        If the m (measure) value is not set, it defaults to NoData."""
    shapeType = POINTM
    pointShape = Shape(shapeType)
    pointShape.points.append([x, y, m])
    self.shape(pointShape)