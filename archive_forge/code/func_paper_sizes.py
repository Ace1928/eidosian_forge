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
def paper_sizes():
    """Known paper formats @ 72 dpi as a dictionary. Key is the format string
    like "a4" for ISO-A4. Value is the tuple (width, height).

    Information taken from the following web sites:
    www.din-formate.de
    www.din-formate.info/amerikanische-formate.html
    www.directtools.de/wissen/normen/iso.htm
    """
    return {'a0': (2384, 3370), 'a1': (1684, 2384), 'a10': (74, 105), 'a2': (1191, 1684), 'a3': (842, 1191), 'a4': (595, 842), 'a5': (420, 595), 'a6': (298, 420), 'a7': (210, 298), 'a8': (147, 210), 'a9': (105, 147), 'b0': (2835, 4008), 'b1': (2004, 2835), 'b10': (88, 125), 'b2': (1417, 2004), 'b3': (1001, 1417), 'b4': (709, 1001), 'b5': (499, 709), 'b6': (354, 499), 'b7': (249, 354), 'b8': (176, 249), 'b9': (125, 176), 'c0': (2599, 3677), 'c1': (1837, 2599), 'c10': (79, 113), 'c2': (1298, 1837), 'c3': (918, 1298), 'c4': (649, 918), 'c5': (459, 649), 'c6': (323, 459), 'c7': (230, 323), 'c8': (162, 230), 'c9': (113, 162), 'card-4x6': (288, 432), 'card-5x7': (360, 504), 'commercial': (297, 684), 'executive': (522, 756), 'invoice': (396, 612), 'ledger': (792, 1224), 'legal': (612, 1008), 'legal-13': (612, 936), 'letter': (612, 792), 'monarch': (279, 540), 'tabloid-extra': (864, 1296)}