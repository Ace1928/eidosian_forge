from __future__ import annotations
import re
from functools import lru_cache
from itertools import chain, count
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from fontTools import ttLib
from fontTools.subset.util import _add_method
from fontTools.ttLib.tables.S_V_G_ import SVGDocument
def group_elements_by_id(tree: etree.Element) -> Dict[str, etree.Element]:
    return {el.attrib['id']: el for el in xpath('//svg:*[@id]')(tree)}