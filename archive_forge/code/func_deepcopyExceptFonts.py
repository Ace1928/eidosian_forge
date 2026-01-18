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
def deepcopyExceptFonts(self):
    """Allow deep-copying a DesignSpace document without deep-copying
        attached UFO fonts or TTFont objects. The :attr:`font` attribute
        is shared by reference between the original and the copy.

        .. versionadded:: 5.0
        """
    fonts = [source.font for source in self.sources]
    try:
        for source in self.sources:
            source.font = None
        res = copy.deepcopy(self)
        for source, font in zip(res.sources, fonts):
            source.font = font
        return res
    finally:
        for source, font in zip(self.sources, fonts):
            source.font = font