from __future__ import annotations
import xml.etree.ElementTree as etree
from typing import TYPE_CHECKING, Iterable, Any
from . import util
def isstate(self, state: Any) -> bool:
    """ Test that top (current) level is of given state. """
    if len(self):
        return self[-1] == state
    else:
        return False