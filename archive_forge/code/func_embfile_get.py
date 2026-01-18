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
def embfile_get(self, item: typing.Union[int, str]) -> bytes:
    """Get the content of an item in the EmbeddedFiles array.

        Args:
            item: number or name of item.
        Returns:
            (bytes) The file content.
        """
    idx = self._embeddedFileIndex(item)
    return self._embeddedFileGet(idx)