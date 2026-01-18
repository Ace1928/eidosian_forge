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
def findDefault(self):
    """Set and return SourceDescriptor at the default location or None.

        The default location is the set of all `default` values in user space
        of all axes.

        This function updates the document's :attr:`default` value.

        .. versionchanged:: 5.0
           Allow the default source to not specify some of the axis values, and
           they are assumed to be the default.
           See :meth:`SourceDescriptor.getFullDesignLocation()`
        """
    self.default = None
    defaultDesignLocation = self.newDefaultLocation()
    for sourceDescriptor in self.sources:
        if sourceDescriptor.getFullDesignLocation(self) == defaultDesignLocation:
            self.default = sourceDescriptor
            return sourceDescriptor
    return None