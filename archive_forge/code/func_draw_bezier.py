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
def draw_bezier(self, p1: point_like, p2: point_like, p3: point_like, p4: point_like):
    """Draw a standard cubic Bezier curve."""
    p1 = Point(p1)
    p2 = Point(p2)
    p3 = Point(p3)
    p4 = Point(p4)
    if not self.last_point == p1:
        self.draw_cont += '%g %g m\n' % JM_TUPLE(p1 * self.ipctm)
    self.draw_cont += '%g %g %g %g %g %g c\n' % JM_TUPLE(list(p2 * self.ipctm) + list(p3 * self.ipctm) + list(p4 * self.ipctm))
    self.updateRect(p1)
    self.updateRect(p2)
    self.updateRect(p3)
    self.updateRect(p4)
    self.last_point = p4
    return self.last_point