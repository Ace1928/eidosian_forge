from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def coerceElement(self, name):
    return self.toXmlName(name)