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
def multipatch(self, parts, partTypes):
    """Creates a MULTIPATCH shape.
        Parts is a collection of 3D surface patches, each made up of a list of xyzm values.
        PartTypes is a list of types that define each of the surface patches.
        The types can be any of the following module constants: TRIANGLE_STRIP,
        TRIANGLE_FAN, OUTER_RING, INNER_RING, FIRST_RING, or RING.
        If the z (elevation) value is not included, it defaults to 0.
        If the m (measure) value is not included, it defaults to None (NoData)."""
    shapeType = MULTIPATCH
    polyShape = Shape(shapeType)
    polyShape.parts = []
    polyShape.points = []
    for part in parts:
        polyShape.parts.append(len(polyShape.points))
        for point in part:
            if not isinstance(point, list):
                point = list(point)
            polyShape.points.append(point)
    polyShape.partTypes = partTypes
    self.shape(polyShape)