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
def draw_quad(self, quad: quad_like):
    """Draw a Quad."""
    q = Quad(quad)
    return self.draw_polyline([q.ul, q.ll, q.lr, q.ur, q.ul])