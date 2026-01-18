from __future__ import absolute_import, division, unicode_literals
from six import text_type
import re
from copy import copy
from . import base
from .. import _ihatexml
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
def hasContent(self):
    """Return true if the node has children or text"""
    return bool(self._element.text or len(self._element))