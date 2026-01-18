from __future__ import annotations
from datetime import datetime
from . import util
import re
from . import css_types as ct
import unicodedata
import bs4  # type: ignore[import]
from typing import Iterator, Iterable, Any, Callable, Sequence, cast  # noqa: F401
def match_nth(self, el: bs4.Tag, nth: bs4.Tag) -> bool:
    """Match `nth` elements."""
    matched = True
    for n in nth:
        matched = False
        if n.selectors and (not self.match_selectors(el, n.selectors)):
            break
        parent = self.get_parent(el)
        if parent is None:
            parent = self.create_fake_parent(el)
        last = n.last
        last_index = len(parent) - 1
        index = last_index if last else 0
        relative_index = 0
        a = n.a
        b = n.b
        var = n.n
        count = 0
        count_incr = 1
        factor = -1 if last else 1
        idx = last_idx = a * count + b if var else a
        if var:
            adjust = None
            while idx < 1 or idx > last_index:
                if idx < 0:
                    diff_low = 0 - idx
                    if adjust is not None and adjust == 1:
                        break
                    adjust = -1
                    count += count_incr
                    idx = last_idx = a * count + b if var else a
                    diff = 0 - idx
                    if diff >= diff_low:
                        break
                else:
                    diff_high = idx - last_index
                    if adjust is not None and adjust == -1:
                        break
                    adjust = 1
                    count += count_incr
                    idx = last_idx = a * count + b if var else a
                    diff = idx - last_index
                    if diff >= diff_high:
                        break
                    diff_high = diff
            lowest = count
            if a < 0:
                while idx >= 1:
                    lowest = count
                    count += count_incr
                    idx = last_idx = a * count + b if var else a
                count_incr = -1
            count = lowest
            idx = last_idx = a * count + b if var else a
        while 1 <= idx <= last_index + 1:
            child = None
            for child in self.get_children(parent, start=index, reverse=factor < 0, tags=False):
                index += factor
                if not self.is_tag(child):
                    continue
                if n.selectors and (not self.match_selectors(child, n.selectors)):
                    continue
                if n.of_type and (not self.match_nth_tag_type(el, child)):
                    continue
                relative_index += 1
                if relative_index == idx:
                    if child is el:
                        matched = True
                    else:
                        break
                if child is el:
                    break
            if child is el:
                break
            last_idx = idx
            count += count_incr
            if count < 0:
                break
            idx = a * count + b if var else a
            if last_idx == idx:
                break
        if not matched:
            break
    return matched