import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def find_rel_links(self, rel):
    """
        Find any links like ``<a rel="{rel}">...</a>``; returns a list of elements.
        """
    rel = rel.lower()
    return [el for el in _rel_links_xpath(self) if el.get('rel').lower() == rel]