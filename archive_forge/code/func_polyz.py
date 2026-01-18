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
def polyz(self, polys):
    """Creates a POLYGONZ shape.
        Polys is a collection of polygons, each made up of a list of xyzm values.
        Note that for ordinary polygons the coordinates must run in a clockwise direction.
        If some of the polygons are holes, these must run in a counterclockwise direction.
        If the z (elevation) value is not included, it defaults to 0.
        If the m (measure) value is not included, it defaults to None (NoData)."""
    shapeType = POLYGONZ
    self._shapeparts(parts=polys, shapeType=shapeType)