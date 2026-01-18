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
def args_match(args, *types):
    """
    Returns true if <args> matches <types>.

    Each item in <types> is a type or tuple of types. Any of these types will
    match an item in <args>. `None` will match anything in <args>. `type(None)`
    will match an arg whose value is `None`.
    """
    j = 0
    for i in range(len(types)):
        type_ = types[i]
        if j >= len(args):
            if isinstance(type_, tuple) and None in type_:
                continue
            else:
                return False
        if type_ is not None and (not isinstance(args[j], type_)):
            return False
        j += 1
    if j != len(args):
        return False
    return True