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
def jm_lineart_pop_clip(dev, ctx):
    if not dev.clips or not dev.scissors:
        return
    len_ = len(dev.scissors)
    if len_ < 1:
        return
    del dev.scissors[-1]
    dev.depth -= 1