from collections import defaultdict as _defaultdict
from collections.abc import Mapping
import os
from toolz.dicttoolz import (merge, merge_with, valmap, keymap, update_in,
from toolz.functoolz import identity
from toolz.utils import raises
def test_merge_with_non_dict_mappings():

    class Foo(Mapping):

        def __init__(self, d):
            self.d = d

        def __iter__(self):
            return iter(self.d)

        def __getitem__(self, key):
            return self.d[key]

        def __len__(self):
            return len(self.d)
    d = Foo({1: 1})
    assert merge(d) is d or merge(d) == {1: 1}
    assert merge_with(sum, d) == {1: 1}