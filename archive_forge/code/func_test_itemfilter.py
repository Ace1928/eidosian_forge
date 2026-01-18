from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_itemfilter(self):
    D, kw = (self.D, self.kw)
    assert itemfilter(lambda item: iseven(item[0]), D({1: 2, 2: 3}), **kw) == D({2: 3})
    assert itemfilter(lambda item: iseven(item[1]), D({1: 2, 2: 3}), **kw) == D({1: 2})