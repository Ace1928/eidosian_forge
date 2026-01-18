import pytest
from textwrap import dedent
import platform
import srsly
from .roundtrip import (
def test_merge_00(self):
    data = load(self.merge_yaml)
    d = data[4]
    ok = True
    for k in d:
        for o in [5, 6, 7]:
            x = d.get(k)
            y = data[o].get(k)
            if not isinstance(x, int):
                x = x.split('/')[0]
                y = y.split('/')[0]
            if x != y:
                ok = False
                print('key', k, d.get(k), data[o].get(k))
    assert ok