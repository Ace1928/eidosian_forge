from __future__ import annotations
from typing import TYPE_CHECKING
from . import Extension
from ..treeprocessors import Treeprocessor
import re
def sanitize_name(self, name: str) -> str:
    """
        Sanitize name as 'an XML Name, minus the `:`.'
        See <https://www.w3.org/TR/REC-xml-names/#NT-NCName>.
        """
    return self.NAME_RE.sub('_', name)