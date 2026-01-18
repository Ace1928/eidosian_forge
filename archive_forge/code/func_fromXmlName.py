from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def fromXmlName(self, name):
    for item in set(self.replacementRegexp.findall(name)):
        name = name.replace(item, self.unescapeChar(item))
    return name