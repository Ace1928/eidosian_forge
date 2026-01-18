from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def test_nested_tasks(self):
    d = {'x': 1, 'y': (inc, 'x'), 'z': (add, (inc, 'x'), 'y')}
    assert self.get(d, 'z') == 4