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
def readVariableFonts(self):
    if self.documentObject.formatTuple < (5, 0):
        return
    xml_attrs = {'name', 'filename'}
    for variableFontElement in self.root.findall('.variable-fonts/variable-font'):
        unknown_attrs = set(variableFontElement.attrib) - xml_attrs
        if unknown_attrs:
            raise DesignSpaceDocumentError(f'variable-font element contains unknown attributes: {', '.join(unknown_attrs)}')
        name = variableFontElement.get('name')
        if name is None:
            raise DesignSpaceDocumentError('variable-font element must have a name attribute.')
        filename = variableFontElement.get('filename')
        axisSubsetsElement = variableFontElement.find('.axis-subsets')
        if axisSubsetsElement is None:
            raise DesignSpaceDocumentError('variable-font element must contain an axis-subsets element.')
        axisSubsets = []
        for axisSubset in axisSubsetsElement.iterfind('.axis-subset'):
            axisSubsets.append(self.readAxisSubset(axisSubset))
        lib = None
        libElement = variableFontElement.find('.lib')
        if libElement is not None:
            lib = plistlib.fromtree(libElement[0])
        variableFont = self.variableFontsDescriptorClass(name=name, filename=filename, axisSubsets=axisSubsets, lib=lib)
        self.documentObject.variableFonts.append(variableFont)