import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def find_class(self, class_name):
    """
        Find any elements with the given class name.
        """
    return _class_xpath(self, class_name=class_name)