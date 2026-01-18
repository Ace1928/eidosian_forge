import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def planish_line(p1: point_like, p2: point_like) -> Matrix:
    """Compute matrix which maps line from p1 to p2 to the x-axis, such that it
    maintains its length and p1 * matrix = Point(0, 0).

    Args:
        p1, p2: point_like
    Returns:
        Matrix which maps p1 to Point(0, 0) and p2 to a point on the x axis at
        the same distance to Point(0,0). Will always combine a rotation and a
        transformation.
    """
    p1 = Point(p1)
    p2 = Point(p2)
    return Matrix(util_hor_matrix(p1, p2))