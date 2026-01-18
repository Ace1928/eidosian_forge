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
def distance_to(self, *args):
    """Return distance to rectangle or another point."""
    if not len(args) > 0:
        raise ValueError('at least one parameter must be given')
    x = args[0]
    if len(x) == 2:
        x = Point(x)
    elif len(x) == 4:
        x = Rect(x)
    else:
        raise ValueError('arg1 must be point-like or rect-like')
    if len(args) > 1:
        unit = args[1]
    else:
        unit = 'px'
    u = {'px': (1.0, 1.0), 'in': (1.0, 72.0), 'cm': (2.54, 72.0), 'mm': (25.4, 72.0)}
    f = u[unit][0] / u[unit][1]
    if type(x) is Point:
        return abs(self - x) * f
    r = Rect(x.top_left, x.top_left)
    r = r | x.bottom_right
    if self in r:
        return 0.0
    if self.x > r.x1:
        if self.y >= r.y1:
            return self.distance_to(r.bottom_right, unit)
        elif self.y <= r.y0:
            return self.distance_to(r.top_right, unit)
        else:
            return (self.x - r.x1) * f
    elif r.x0 <= self.x <= r.x1:
        if self.y >= r.y1:
            return (self.y - r.y1) * f
        else:
            return (r.y0 - self.y) * f
    elif self.y >= r.y1:
        return self.distance_to(r.bottom_left, unit)
    elif self.y <= r.y0:
        return self.distance_to(r.top_left, unit)
    else:
        return (r.x0 - self.x) * f