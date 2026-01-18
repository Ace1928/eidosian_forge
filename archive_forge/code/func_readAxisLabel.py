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
def readAxisLabel(self, element: ET.Element):
    xml_attrs = {'userminimum', 'uservalue', 'usermaximum', 'name', 'elidable', 'oldersibling', 'linkeduservalue'}
    unknown_attrs = set(element.attrib) - xml_attrs
    if unknown_attrs:
        raise DesignSpaceDocumentError(f'label element contains unknown attributes: {', '.join(unknown_attrs)}')
    name = element.get('name')
    if name is None:
        raise DesignSpaceDocumentError('label element must have a name attribute.')
    valueStr = element.get('uservalue')
    if valueStr is None:
        raise DesignSpaceDocumentError('label element must have a uservalue attribute.')
    value = float(valueStr)
    minimumStr = element.get('userminimum')
    minimum = float(minimumStr) if minimumStr is not None else None
    maximumStr = element.get('usermaximum')
    maximum = float(maximumStr) if maximumStr is not None else None
    linkedValueStr = element.get('linkeduservalue')
    linkedValue = float(linkedValueStr) if linkedValueStr is not None else None
    elidable = True if element.get('elidable') == 'true' else False
    olderSibling = True if element.get('oldersibling') == 'true' else False
    labelNames = {lang: label_name.text or '' for label_name in element.findall('labelname') for attr, lang in label_name.items() if attr == XML_LANG}
    return self.axisLabelDescriptorClass(name=name, userValue=value, userMinimum=minimum, userMaximum=maximum, elidable=elidable, olderSibling=olderSibling, linkedUserValue=linkedValue, labelNames=labelNames)