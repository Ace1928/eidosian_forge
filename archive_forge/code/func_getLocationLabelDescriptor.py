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
def getLocationLabelDescriptor(self, doc: 'DesignSpaceDocument') -> Optional[LocationLabelDescriptor]:
    """Get the :class:`LocationLabelDescriptor` instance that matches
        this instances's :attr:`locationLabel`.

        Raises if the named label can't be found.

        .. versionadded:: 5.0
        """
    if self.locationLabel is None:
        return None
    label = doc.getLocationLabel(self.locationLabel)
    if label is None:
        raise DesignSpaceDocumentError(f'InstanceDescriptor.getLocationLabelDescriptor(): unknown location label `{self.locationLabel}` in instance `{self.name}`.')
    return label