from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def test_badkey(self):
    d = {':x': 1, ':y': (inc, ':x'), ':z': (add, ':x', ':y')}
    try:
        result = self.get(d, 'badkey')
    except KeyError:
        pass
    else:
        msg = 'Expected `{}` with badkey to raise KeyError.\n'
        msg += f"Obtained '{result}' instead."
        assert False, msg.format(self.get.__name__)