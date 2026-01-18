import functools
import os.path
from .errors import DistutilsFileError
from .py39compat import zip_strict
from ._functools import splat
def missing_as_newer(source):
    return missing == 'newer' and (not os.path.exists(source))