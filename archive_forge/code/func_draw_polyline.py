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
def draw_polyline(self, points: list):
    """Draw several connected line segments."""
    for i, p in enumerate(points):
        if i == 0:
            if not self.last_point == Point(p):
                self.draw_cont += '%g %g m\n' % JM_TUPLE(Point(p) * self.ipctm)
                self.last_point = Point(p)
        else:
            self.draw_cont += '%g %g l\n' % JM_TUPLE(Point(p) * self.ipctm)
        self.updateRect(p)
    self.last_point = Point(points[-1])
    return self.last_point