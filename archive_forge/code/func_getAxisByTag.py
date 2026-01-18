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
def getAxisByTag(self, tag: str) -> AxisDescriptor | DiscreteAxisDescriptor | None:
    """Return the axis with the given ``tag``, or ``None`` if no such axis exists."""
    return next((axis for axis in self.axes if axis.tag == tag), None)