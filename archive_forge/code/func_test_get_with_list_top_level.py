from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def test_get_with_list_top_level(self):
    d = {'a': [1, 2, 3], 'b': 'a', 'c': [1, (inc, 1)], 'd': [(sum, 'a')], 'e': ['a', 'b'], 'f': [[[(sum, 'a'), 'c'], (sum, 'b')], 2]}
    assert self.get(d, 'a') == [1, 2, 3]
    assert self.get(d, 'b') == [1, 2, 3]
    assert self.get(d, 'c') == [1, 2]
    assert self.get(d, 'd') == [6]
    assert self.get(d, 'e') == [[1, 2, 3], [1, 2, 3]]
    assert self.get(d, 'f') == [[[6, [1, 2]], 6], 2]