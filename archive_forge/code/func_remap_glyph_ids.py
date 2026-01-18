from __future__ import annotations
import re
from functools import lru_cache
from itertools import chain, count
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple
from fontTools import ttLib
from fontTools.subset.util import _add_method
from fontTools.ttLib.tables.S_V_G_ import SVGDocument
def remap_glyph_ids(svg: etree.Element, glyph_index_map: Dict[int, int]) -> Dict[str, str]:
    elements = group_elements_by_id(svg)
    id_map = {}
    for el_id, el in elements.items():
        m = GID_RE.match(el_id)
        if not m:
            continue
        old_index = int(m.group(1))
        new_index = glyph_index_map.get(old_index)
        if new_index is not None:
            if old_index == new_index:
                continue
            new_id = f'glyph{new_index}'
        else:
            new_id = f'.{el_id}'
            n = count(1)
            while new_id in elements:
                new_id = f'{new_id}.{next(n)}'
        id_map[el_id] = new_id
        el.attrib['id'] = new_id
    return id_map