from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_keymap(self):
    D, kw = (self.D, self.kw)
    assert keymap(inc, D({1: 1, 2: 2}), **kw) == D({2: 1, 3: 2})