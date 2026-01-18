from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def isHTMLIntegrationPoint(self, element):
    if element.name == 'annotation-xml' and element.namespace == namespaces['mathml']:
        return 'encoding' in element.attributes and element.attributes['encoding'].translate(asciiUpper2Lower) in ('text/html', 'application/xhtml+xml')
    else:
        return (element.namespace, element.name) in htmlIntegrationPointElements