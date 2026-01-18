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
@contextlib.contextmanager
def readonly_cursor(self):
    cursor = self.session.cursor()
    try:
        yield cursor
    finally:
        cursor.close()
        self.session.rollback()