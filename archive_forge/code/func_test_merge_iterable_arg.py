from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_merge_iterable_arg(self):
    D, kw = (self.D, self.kw)
    assert merge([D({1: 1, 2: 2}), D({3: 4})], **kw) == D({1: 1, 2: 2, 3: 4})