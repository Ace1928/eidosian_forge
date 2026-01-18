import re
import sys
from weakref import ReferenceType
import weakref
from debian._util import resolve_ref, _strI
from debian._deb822_repro._util import BufferingIterator
def clear_parent_if_parent(self, parent):
    if parent is self.parent_element:
        self._parent_element = None