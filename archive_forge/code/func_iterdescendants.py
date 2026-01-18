from __future__ import print_function, absolute_import
import threading
import warnings
from lxml import etree as _etree
from .common import DTDForbidden, EntitiesForbidden, NotSupportedError
def iterdescendants(self, tag=None, *tags):
    iterator = super(RestrictedElement, self).iterdescendants(*tags, tag=tag)
    return self._filter(iterator)