from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_merge_with_iterable_arg(self):
    D, kw = (self.D, self.kw)
    dicts = (D({1: 1, 2: 2}), D({1: 10, 2: 20}))
    assert merge_with(sum, *dicts, **kw) == D({1: 11, 2: 22})
    assert merge_with(sum, dicts, **kw) == D({1: 11, 2: 22})
    assert merge_with(sum, iter(dicts), **kw) == D({1: 11, 2: 22})