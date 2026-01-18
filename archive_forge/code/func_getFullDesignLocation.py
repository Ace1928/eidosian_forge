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
def getFullDesignLocation(self, doc: 'DesignSpaceDocument') -> AnisotropicLocationDict:
    """Get the complete design location of this instance, by combining data
        from the various location fields, default axis values and mappings, and
        top-level location labels.

        The source of truth for this instance's location is determined for each
        axis independently by taking the first not-None field in this list:

        - ``locationLabel``: the location along this axis is the same as the
          matching STAT format 4 label. No anisotropy.
        - ``designLocation[axisName]``: the explicit design location along this
          axis, possibly anisotropic.
        - ``userLocation[axisName]``: the explicit user location along this
          axis. No anisotropy.
        - ``axis.default``: default axis value. No anisotropy.

        .. versionadded:: 5.0
        """
    label = self.getLocationLabelDescriptor(doc)
    if label is not None:
        return doc.map_forward(label.userLocation)
    result: AnisotropicLocationDict = {}
    for axis in doc.axes:
        if axis.name in self.designLocation:
            result[axis.name] = self.designLocation[axis.name]
        elif axis.name in self.userLocation:
            result[axis.name] = axis.map_forward(self.userLocation[axis.name])
        else:
            result[axis.name] = axis.map_forward(axis.default)
    return result