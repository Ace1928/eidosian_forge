import copy
import re
from collections.abc import MutableMapping, MutableSet
from functools import partial
from urllib.parse import urljoin
from .. import etree
from . import defs
from ._setmixin import SetMixin
def drop_tree(self):
    """
        Removes this element from the tree, including its children and
        text.  The tail text is joined to the previous element or
        parent.
        """
    parent = self.getparent()
    assert parent is not None
    if self.tail:
        previous = self.getprevious()
        if previous is None:
            parent.text = (parent.text or '') + self.tail
        else:
            previous.tail = (previous.tail or '') + self.tail
    parent.remove(self)