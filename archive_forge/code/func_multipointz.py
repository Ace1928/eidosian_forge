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
def multipointz(self, points):
    """Creates a MULTIPOINTZ shape.
        Points is a list of xyzm values.
        If the z (elevation) value is not included, it defaults to 0.
        If the m (measure) value is not included, it defaults to None (NoData)."""
    shapeType = MULTIPOINTZ
    points = [points]
    self._shapeparts(parts=points, shapeType=shapeType)