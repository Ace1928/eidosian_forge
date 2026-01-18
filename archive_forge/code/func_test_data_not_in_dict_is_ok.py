from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def test_data_not_in_dict_is_ok(self):
    d = {'x': 1, 'y': (add, 'x', 10)}
    assert self.get(d, 'y') == 11