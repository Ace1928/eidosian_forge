from __future__ import absolute_import, division, unicode_literals
import warnings
import re
import sys
from . import base
from ..constants import DataLossWarning
from .. import constants
from . import etree as etree_builders
from .. import _ihatexml
import lxml.etree as etree
from six import PY3, binary_type
def insertText(self, data, insertBefore=None):
    data = infosetFilter.coerceCharacters(data)
    builder.Element.insertText(self, data, insertBefore)