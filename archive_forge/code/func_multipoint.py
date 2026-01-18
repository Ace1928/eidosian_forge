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
def multipoint(self, points):
    """Creates a MULTIPOINT shape.
        Points is a list of xy values."""
    shapeType = MULTIPOINT
    points = [points]
    self._shapeparts(parts=points, shapeType=shapeType)