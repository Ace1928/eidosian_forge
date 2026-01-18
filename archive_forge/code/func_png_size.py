import contextlib
import getpass
import logging
import os
import sqlite3
import tempfile
import warnings
from io import BytesIO
from os.path import join as pjoin
import numpy
from nibabel.optpkg import optional_package
from .nifti1 import Nifti1Header
def png_size(self, index=None, scale_to_slice=True):
    return len(self.as_png(index=index, scale_to_slice=scale_to_slice))