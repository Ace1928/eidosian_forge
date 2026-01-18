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
def jm_lineart_ignore_text(dev, text, ctm):
    jm_trace_text(dev, text, 3, ctm, None, None, 1, dev.seqno)
    dev.seqno += 1