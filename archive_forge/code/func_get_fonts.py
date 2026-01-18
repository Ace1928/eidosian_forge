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
def get_fonts(self, full=False):
    """List of fonts defined in the page object."""
    CheckParent(self)
    return self.parent.get_page_fonts(self.number, full=full)