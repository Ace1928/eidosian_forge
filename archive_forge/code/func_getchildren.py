from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
def getchildren(self):
    iterator = super(RestrictedElement, self).__iter__()
    return list(self._filter(iterator))