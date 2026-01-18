from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
def posixpath_property(private_name):
    """Generate a propery that holds a path always using forward slashes."""

    def getter(self):
        return getattr(self, private_name)

    def setter(self, value):
        if value is not None:
            value = posix(value)
        setattr(self, private_name, value)
    return property(getter, setter)