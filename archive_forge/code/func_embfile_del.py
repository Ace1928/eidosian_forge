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
def embfile_del(self, item: typing.Union[int, str]):
    """Delete an entry from EmbeddedFiles.

        Notes:
            The argument must be name or index of an EmbeddedFiles item.
            Physical deletion of data will happen on save to a new
            file with appropriate garbage option.
        Args:
            item: name or number of item.
        Returns:
            None
        """
    idx = self._embeddedFileIndex(item)
    return self._embfile_del(idx)