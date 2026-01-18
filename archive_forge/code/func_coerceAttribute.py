from __future__ import absolute_import, division, unicode_literals
import re
import warnings
from .constants import DataLossWarning
def coerceAttribute(self, name, namespace=None):
    if self.dropXmlnsLocalName and name.startswith('xmlns:'):
        warnings.warn('Attributes cannot begin with xmlns', DataLossWarning)
        return None
    elif self.dropXmlnsAttrNs and namespace == 'http://www.w3.org/2000/xmlns/':
        warnings.warn('Attributes cannot be in the xml namespace', DataLossWarning)
        return None
    else:
        return self.toXmlName(name)