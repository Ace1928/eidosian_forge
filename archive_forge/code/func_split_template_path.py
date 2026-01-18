import importlib.util
import os
import posixpath
import sys
import typing as t
import weakref
import zipimport
from collections import abc
from hashlib import sha1
from importlib import import_module
from types import ModuleType
from .exceptions import TemplateNotFound
from .utils import internalcode
def split_template_path(template: str) -> t.List[str]:
    """Split a path into segments and perform a sanity check.  If it detects
    '..' in the path it will raise a `TemplateNotFound` error.
    """
    pieces = []
    for piece in template.split('/'):
        if os.path.sep in piece or (os.path.altsep and os.path.altsep in piece) or piece == os.path.pardir:
            raise TemplateNotFound(template)
        elif piece and piece != '.':
            pieces.append(piece)
    return pieces