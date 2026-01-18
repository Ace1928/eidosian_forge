from __future__ import print_function, unicode_literals
import typing
from typing import IO, cast
import os
import six
import tarfile
from collections import OrderedDict
from . import errors
from ._url_tools import url_quote
from .base import FS
from .compress import write_tar
from .enums import ResourceType
from .errors import IllegalBackReference, NoURL
from .info import Info
from .iotools import RawWrapper
from .opener import open_fs
from .path import basename, frombase, isbase, normpath, parts, relpath
from .permissions import Permissions
from .wrapfs import WrapFS
def delegate_fs(self):
    return self._temp_fs