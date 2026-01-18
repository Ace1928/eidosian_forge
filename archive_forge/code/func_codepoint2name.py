from __future__ import annotations
from . import util
from typing import TYPE_CHECKING, Any, Collection, NamedTuple
import re
import xml.etree.ElementTree as etree
from html import entities
def codepoint2name(code: int) -> str:
    """Return entity definition by code, or the code if not defined."""
    entity = entities.codepoint2name.get(code)
    if entity:
        return '{}{};'.format(util.AMP_SUBSTITUTE, entity)
    else:
        return '%s#%d;' % (util.AMP_SUBSTITUTE, code)