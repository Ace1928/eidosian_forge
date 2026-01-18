import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
def rotated(self, radians):
    """
        Return the axes-aligned bounding box that bounds the result of rotating
        this `Bbox` by an angle of *radians*.
        """
    corners = self.corners()
    corners_rotated = Affine2D().rotate(radians).transform(corners)
    bbox = Bbox.unit()
    bbox.update_from_data_xy(corners_rotated, ignore=True)
    return bbox