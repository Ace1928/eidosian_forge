from __future__ import absolute_import, print_function, unicode_literals
import typing
import abc
import hashlib
import itertools
import os
import six
import threading
import time
import warnings
from contextlib import closing
from functools import partial, wraps
from . import copy, errors, fsencode, iotools, tools, walk, wildcard
from .copy import copy_modified_time
from .glob import BoundGlobber
from .mode import validate_open_mode
from .path import abspath, join, normpath
from .time import datetime_to_epoch
from .walk import Walker
def getospath(self, path):
    """Get the *system path* to a resource, in the OS' prefered encoding.

        Arguments:
            path (str): A path on the filesystem.

        Returns:
            str: the *system path* of the resource, if any.

        Raises:
            fs.errors.NoSysPath: If there is no corresponding system path.

        This method takes the output of `~getsyspath` and encodes it to
        the filesystem's prefered encoding. In Python3 this step is
        not required, as the `os` module will do it automatically. In
        Python2.7, the encoding step is required to support filenames
        on the filesystem that don't encode correctly.

        Note:
            If you want your code to work in Python2.7 and Python3 then
            use this method if you want to work with the OS filesystem
            outside of the OSFS interface.

        """
    syspath = self.getsyspath(path)
    ospath = fsencode(syspath)
    return ospath