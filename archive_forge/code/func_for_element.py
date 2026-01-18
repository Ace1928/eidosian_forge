import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
@for_element.deleter
def for_element(self):
    attrib = self.attrib
    if 'id' in attrib:
        del attrib['id']