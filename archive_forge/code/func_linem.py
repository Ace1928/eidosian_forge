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
def linem(self, lines):
    """Creates a POLYLINEM shape.
        Lines is a collection of lines, each made up of a list of xym values.
        If the m (measure) value is not included, it defaults to None (NoData)."""
    shapeType = POLYLINEM
    self._shapeparts(parts=lines, shapeType=shapeType)