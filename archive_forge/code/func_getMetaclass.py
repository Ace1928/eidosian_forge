from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def getMetaclass(use_metaclass, metaclass_func):
    if use_metaclass:
        return method_decorator_metaclass(metaclass_func)
    else:
        return type