import atexit
import functools
import hashlib
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
import weakref
import numpy as np
from PIL import Image
import matplotlib as mpl
from matplotlib import cbook
from matplotlib.testing.exceptions import ImageComparisonFailure
def make_test_filename(fname, purpose):
    """
    Make a new filename by inserting *purpose* before the file's extension.
    """
    base, ext = os.path.splitext(fname)
    return f'{base}-{purpose}{ext}'