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
@staticmethod
def horizontal_angle(C, P):
    """Return the angle to the horizontal for the connection from C to P.
        This uses the arcus sine function and resolves its inherent ambiguity by
        looking up in which quadrant vector S = P - C is located.
        """
    S = Point(P - C).unit
    alfa = math.asin(abs(S.y))
    if S.x < 0:
        if S.y <= 0:
            alfa = -(math.pi - alfa)
        else:
            alfa = math.pi - alfa
    elif S.y >= 0:
        pass
    else:
        alfa = -alfa
    return alfa