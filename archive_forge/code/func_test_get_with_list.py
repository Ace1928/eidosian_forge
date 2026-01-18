from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def test_get_with_list(self):
    d = {'x': 1, 'y': 2, 'z': (sum, ['x', 'y'])}
    assert self.get(d, ['x', 'y']) == (1, 2)
    assert self.get(d, 'z') == 3