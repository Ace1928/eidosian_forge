import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
@multiple.setter
def multiple(self, value):
    if value:
        self.set('multiple', '')
    elif 'multiple' in self.attrib:
        del self.attrib['multiple']