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
def pointz(self, x, y, z=0, m=None):
    """Creates a POINTZ shape.
        If the z (elevation) value is not set, it defaults to 0.
        If the m (measure) value is not set, it defaults to NoData."""
    shapeType = POINTZ
    pointShape = Shape(shapeType)
    pointShape.points.append([x, y, z, m])
    self.shape(pointShape)