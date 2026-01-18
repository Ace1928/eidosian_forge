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
def linez(self, lines):
    """Creates a POLYLINEZ shape.
        Lines is a collection of lines, each made up of a list of xyzm values.
        If the z (elevation) value is not included, it defaults to 0.
        If the m (measure) value is not included, it defaults to None (NoData)."""
    shapeType = POLYLINEZ
    self._shapeparts(parts=lines, shapeType=shapeType)