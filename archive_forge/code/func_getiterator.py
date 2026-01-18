from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
def getiterator(self, tag=None):
    iterator = super(RestrictedElement, self).getiterator(tag)
    return self._filter(iterator)