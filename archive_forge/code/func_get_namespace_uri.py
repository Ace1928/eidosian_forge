from __future__ import unicode_literals
import sys
import logging
import re
import time
import xml.dom.minidom
from . import __author__, __copyright__, __license__, __version__
from .helpers import TYPE_MAP, TYPE_MARSHAL_FN, TYPE_UNMARSHAL_FN, \
def get_namespace_uri(self, ns):
    """Return the namespace uri for a prefix"""
    element = self._element
    while element is not None and element.attributes is not None:
        try:
            return element.attributes['xmlns:%s' % ns].value
        except KeyError:
            element = element.parentNode