import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
@checked.setter
def checked(self, value):
    if not self.checkable:
        raise AttributeError('Not a checkable input type')
    if value:
        self.set('checked', '')
    else:
        attrib = self.attrib
        if 'checked' in attrib:
            del attrib['checked']