import re
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Tuple
from .lazyre import LazyReCompile
def current_import(cursor_offset: int, line: str) -> Optional[LinePart]:
    baseline = _current_import_re_1.search(line)
    if baseline is None:
        return None
    match1 = _current_import_re_2.search(line[baseline.end():])
    if match1 is None:
        return None
    for m in chain((match1,), _current_import_re_3.finditer(line[baseline.end():])):
        start = baseline.end() + m.start(1)
        end = baseline.end() + m.end(1)
        if start < cursor_offset <= end:
            return LinePart(start, end, m.group(1))
    return None